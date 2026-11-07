"""
EIP-161 emptiness must be judged by an account's fields alone.

A zero-value CALL to an absent precompile runs its code without creating
the account, so the address stays dead. A client that records the account
on that first touch and later reads its own bookkeeping as proof of
existence would skip new-account charges. These tests probe that
execution-side possibility around the shape reported in
https://github.com/erigontech/erigon/issues/23670, where the touch
bookkeeping leaked into BAL construction while state roots agreed.
"""

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
    CodeGasMeasure,
    Conditional,
    Fork,
    Op,
    StateTestFiller,
    Storage,
    Transaction,
)
from execution_testing.forks import Byzantium

from ethereum.crypto.hash import keccak256

from .spec import ref_spec_161

REFERENCE_SPEC_GIT_PATH = ref_spec_161.git_path
REFERENCE_SPEC_VERSION = ref_spec_161.version

# Precompiles that reject empty input. A zero-value call to one of them
# fails, so its touch rolls back with the frame: the same arm then covers
# both a live touch and a reverted one.
REJECTS_EMPTY_INPUT = {Address(0x09), Address(0x0A)} | {
    Address(address) for address in range(0x0B, 0x12)
}

TOUCH_MODES = [
    "none",
    "zero_value_call",
    "failed_value_call",
    "reverted_zero_value_call",
    "live_then_reverted_zero_value_call",
    "reverted_value_call",
    "reverted_selfdestruct",
    "live_then_reverted_selfdestruct",
]

# Touch modes that leave a non-precompile target warm in the outer frame
# under EIP-2929 (Berlin+). Purely reverted child touches roll back their
# warm access along with the touch itself.
WARMING_TOUCH_MODES = {
    "zero_value_call",
    "failed_value_call",
    "live_then_reverted_zero_value_call",
    "live_then_reverted_selfdestruct",
}


def touch_code(
    pre: Alloc,
    fork: Fork,
    target: Address,
    touch: str,
    storage: Storage,
) -> Bytecode:
    """
    Return the same-transaction touch that precedes the probe.

    The touch call's result is stored so a touch that never ran fails
    the fill instead of collapsing the case into its untouched control.
    For reverted child-frame touches, the child frame only halts/reverts
    when the inner touch call returns the expected status, witnessing
    both the inner touch and its outer rollback.
    """
    touch_succeeds = target not in REJECTS_EMPTY_INPUT
    expected_inner = 1 if touch_succeeds else 0
    halt_op = Op.REVERT(0, 0) if fork >= Byzantium else Op.INVALID

    if touch == "none":
        return Bytecode()
    elif touch == "zero_value_call":
        return Op.SSTORE(
            storage.store_next(expected_inner, "touch_call_result"),
            Op.CALL(gas=100_000, address=target),
        )
    elif touch == "failed_value_call":
        # The value exceeds the caller's balance, so the transfer fails
        # after the target access is charged. EIP-161 does not count
        # this as a touch; the case probes a client that records it.
        return Op.SSTORE(
            storage.store_next(0, "touch_call_result"),
            Op.CALL(gas=100_000, address=target, value=2**100),
        )
    elif touch == "reverted_zero_value_call":
        toucher = pre.deploy_contract(
            code=Conditional(
                condition=Op.EQ(
                    Op.CALL(gas=100_000, address=target),
                    expected_inner,
                ),
                if_true=halt_op,
                if_false=Op.STOP,
            )
        )
        return Op.SSTORE(
            storage.store_next(0, "touch_call_result"),
            Op.CALL(gas=500_000, address=toucher),
        )
    elif touch == "live_then_reverted_zero_value_call":
        toucher = pre.deploy_contract(
            code=Conditional(
                condition=Op.EQ(
                    Op.CALL(gas=100_000, address=target),
                    expected_inner,
                ),
                if_true=halt_op,
                if_false=Op.STOP,
            )
        )
        return Op.SSTORE(
            storage.store_next(expected_inner, "live_touch_call_result"),
            Op.CALL(gas=100_000, address=target),
        ) + Op.SSTORE(
            storage.store_next(0, "reverted_touch_call_result"),
            Op.CALL(gas=500_000, address=toucher),
        )
    elif touch == "reverted_value_call":
        toucher = pre.deploy_contract(
            code=Conditional(
                condition=Op.EQ(
                    Op.CALL(gas=100_000, address=target, value=1),
                    expected_inner,
                ),
                if_true=halt_op,
                if_false=Op.STOP,
            ),
            balance=1,
        )
        return Op.SSTORE(
            storage.store_next(0, "touch_call_result"),
            Op.CALL(gas=500_000, address=toucher),
        )
    elif touch == "reverted_selfdestruct":
        zero_destroyer = pre.deploy_contract(code=Op.SELFDESTRUCT(target))
        toucher = pre.deploy_contract(
            code=Conditional(
                condition=Op.EQ(
                    Op.CALL(gas=100_000, address=zero_destroyer),
                    1,
                ),
                if_true=halt_op,
                if_false=Op.STOP,
            )
        )
        return Op.SSTORE(
            storage.store_next(0, "touch_call_result"),
            Op.CALL(gas=500_000, address=toucher),
        )
    elif touch == "live_then_reverted_selfdestruct":
        live_zero_destroyer = pre.deploy_contract(code=Op.SELFDESTRUCT(target))
        reverted_zero_destroyer = pre.deploy_contract(
            code=Op.SELFDESTRUCT(target)
        )
        toucher = pre.deploy_contract(
            code=Conditional(
                condition=Op.EQ(
                    Op.CALL(gas=100_000, address=reverted_zero_destroyer),
                    1,
                ),
                if_true=halt_op,
                if_false=Op.STOP,
            )
        )
        return Op.SSTORE(
            storage.store_next(1, "live_touch_call_result"),
            Op.CALL(gas=100_000, address=live_zero_destroyer),
        ) + Op.SSTORE(
            storage.store_next(0, "reverted_touch_call_result"),
            Op.CALL(gas=500_000, address=toucher),
        )
    else:
        raise ValueError(f"Unknown touch: {touch}")


@pytest.mark.valid_from("ConstantinopleFix")
@pytest.mark.with_all_precompiles
@pytest.mark.parametrize("touch", TOUCH_MODES[1:])
@pytest.mark.parametrize("funded", [False, True])
def test_extcodehash_after_precompile_touch(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    precompile: Address,
    touch: str,
    funded: bool,
) -> None:
    """
    Probe a precompile's fields after touching it in the same transaction.

    The touch must not make an absent precompile look existent, and must
    not make a funded one look absent.
    """
    storage = Storage()
    if funded:
        pre.fund_address(precompile, 1)
    expected_hash = keccak256(b"") if funded else 0
    expected_balance = 1 if funded else 0

    code = (
        Op.SSTORE(
            storage.store_next(expected_hash, "hash_before"),
            Op.EXTCODEHASH(precompile),
        )
        + touch_code(pre, fork, precompile, touch, storage)
        + Op.SSTORE(
            storage.store_next(expected_hash, "hash_after"),
            Op.EXTCODEHASH(precompile),
        )
        + Op.SSTORE(
            storage.store_next(0, "size_after"),
            Op.EXTCODESIZE(precompile),
        )
        + Op.SSTORE(
            storage.store_next(expected_balance, "balance_after"),
            Op.BALANCE(precompile),
        )
    )
    caller = pre.deploy_contract(code, storage=storage.canary())

    tx = Transaction(sender=pre.fund_eoa(), to=caller)

    post = {
        caller: Account(storage=storage),
        precompile: (Account(balance=1) if funded else Account.NONEXISTENT),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("ConstantinopleFix")
@pytest.mark.parametrize("touch", TOUCH_MODES[1:])
@pytest.mark.parametrize("funded", [False, True])
def test_extcodehash_after_empty_account_touch(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    touch: str,
    funded: bool,
) -> None:
    """
    Probe a regular account's fields after touching it in the same
    transaction via live or reverted touches.

    Neither a live zero-value touch nor a reverted child-frame touch
    (zero-value call, value call, or selfdestruct) may make an absent
    account look existent or a funded one look absent.
    """
    storage = Storage()
    target = pre.fund_eoa(amount=1) if funded else pre.nonexistent_account()
    expected_hash = keccak256(b"") if funded else 0
    expected_balance = 1 if funded else 0

    code = (
        Op.SSTORE(
            storage.store_next(expected_hash, "hash_before"),
            Op.EXTCODEHASH(target),
        )
        + touch_code(pre, fork, target, touch, storage)
        + Op.SSTORE(
            storage.store_next(expected_hash, "hash_after"),
            Op.EXTCODEHASH(target),
        )
        + Op.SSTORE(
            storage.store_next(0, "size_after"),
            Op.EXTCODESIZE(target),
        )
        + Op.SSTORE(
            storage.store_next(expected_balance, "balance_after"),
            Op.BALANCE(target),
        )
    )
    caller = pre.deploy_contract(code, storage=storage.canary())

    tx = Transaction(sender=pre.fund_eoa(), to=caller)

    post = {
        caller: Account(storage=storage),
        target: (Account(balance=1) if funded else Account.NONEXISTENT),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("SpuriousDragon")
@pytest.mark.parametrize(
    "precompile,args_size",
    [
        pytest.param(Address(0x01), 0, id="ecrecover"),
        # 15 input words lift the RIPEMD-160 cost above the value
        # stipend, so the callee always runs out of gas.
        pytest.param(Address(0x03), 480, id="ripemd160"),
        pytest.param(None, 0, id="empty_eoa"),
    ],
)
@pytest.mark.parametrize("touch", TOUCH_MODES)
def test_call_new_account_charge_after_precompile_touch(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    precompile: Address | None,
    args_size: int,
    touch: str,
) -> None:
    """
    Charge the full value-call cost to a dead precompile or empty EOA
    that was touched earlier in the transaction.

    For precompiles, the measured call forwards zero gas, so the callee
    runs on the value stipend alone and halts; from Amsterdam the
    new-account charge is state gas and is refunded when the callee
    halts, so a second call creates the account for real and is measured
    too. For an absent EOA, a zero-gas value call succeeds on the
    stipend alone with zero callee execution cost, creating the account
    directly.
    """
    storage = Storage()
    target = (
        precompile if precompile is not None else pre.nonexistent_account()
    )
    address_warm = precompile is not None or touch in WARMING_TOUCH_MODES

    if precompile is None:
        creating_call = Op.CALL(
            gas=0,
            address=target,
            value=1,
            value_transfer=True,
            account_new=True,
            address_warm=address_warm,
        )
        code = touch_code(pre, fork, target, touch, storage) + CodeGasMeasure(
            code=creating_call,
            extra_stack_items=1,
            sstore_key=storage.store_next(
                creating_call.gas_cost(fork) - fork.call_value_stipend(),
                "creating_call_cost",
            ),
        )
    else:
        measured_call = Op.CALL(
            gas=0,
            address=target,
            value=1,
            args_size=args_size,
            value_transfer=True,
            account_new=True,
            address_warm=address_warm,
            new_memory_size=args_size,
        )
        creating_call = Op.CALL(
            gas=100_000,
            address=target,
            value=1,
            value_transfer=True,
            account_new=True,
            address_warm=address_warm,
        )
        # The creating call runs the precompile on empty input, and the
        # stipend the callee received for free comes off the caller's delta.
        gas_costs = fork.gas_costs()
        if precompile == Address(0x01):
            callee_cost = gas_costs.PRECOMPILE_ECRECOVER
        elif precompile == Address(0x03):
            callee_cost = gas_costs.PRECOMPILE_RIPEMD160_BASE
        else:
            raise ValueError(f"Unknown precompile: {precompile}")
        creating_call_cost = (
            creating_call.gas_cost(fork)
            + callee_cost
            - fork.call_value_stipend()
        )

        code = (
            touch_code(pre, fork, target, touch, storage)
            + CodeGasMeasure(
                code=measured_call,
                extra_stack_items=1,
                sstore_key=storage.store_next(
                    measured_call.execution_cost(fork), "measured_call_cost"
                ),
            )
            + CodeGasMeasure(
                code=creating_call,
                extra_stack_items=1,
                sstore_key=storage.store_next(
                    creating_call_cost, "creating_call_cost"
                ),
            )
        )
    caller = pre.deploy_contract(code, balance=1, storage=storage.canary())

    tx = Transaction(
        sender=pre.fund_eoa(),
        to=caller,
        state_gas_reservoir=0,  # Do not hide state gas from Op.GAS
    )

    post = {
        caller: Account(storage=storage, balance=0),
        target: Account(balance=1),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("SpuriousDragon")
@pytest.mark.parametrize(
    "precompile",
    [
        pytest.param(Address(0x01), id="ecrecover"),
        pytest.param(Address(0x03), id="ripemd160"),
        pytest.param(None, id="empty_eoa"),
    ],
)
@pytest.mark.parametrize("touch", TOUCH_MODES)
def test_selfdestruct_beneficiary_charge_after_precompile_touch(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    precompile: Address | None,
    touch: str,
) -> None:
    """
    Charge the beneficiary creation cost when a funded contract
    self-destructs to a dead precompile or empty EOA touched earlier in
    the transaction.

    The measured cost covers the outer call plus the destroyer frame.
    From Amsterdam the beneficiary creation charge splits into execution
    gas, which the GAS delta still sees, and state gas, which the sender
    balance pins in the state root.
    """
    storage = Storage()
    target = (
        precompile if precompile is not None else pre.nonexistent_account()
    )
    address_warm = precompile is not None or touch in WARMING_TOUCH_MODES

    destroyer_code = Op.SELFDESTRUCT(
        target, account_new=True, address_warm=address_warm
    )
    destroyer = pre.deploy_contract(destroyer_code, balance=1)

    outer_call = Op.CALL(gas=100_000, address=destroyer)
    code = touch_code(pre, fork, target, touch, storage) + CodeGasMeasure(
        code=outer_call,
        extra_stack_items=1,
        sstore_key=storage.store_next(
            outer_call.execution_cost(fork)
            + destroyer_code.execution_cost(fork),
            "destroyer_call_cost",
        ),
    )
    caller = pre.deploy_contract(code, storage=storage.canary())

    tx = Transaction(sender=pre.fund_eoa(), to=caller)

    post = {
        caller: Account(storage=storage),
        target: Account(balance=1),
    }
    state_test(pre=pre, post=post, tx=tx)
