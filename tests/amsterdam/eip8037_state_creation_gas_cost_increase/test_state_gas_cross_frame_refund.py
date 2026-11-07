"""
Test where a state gas refund lands when it is credited in a
different frame than the spilled charge it undoes.

A state charge spilled from `gas_left` can be refunded in a child
frame. The credit lands in the child's reservoir and merges upward as
reservoir, which repays the parent's outstanding spill as the child
merges, at any depth and through call and create opcodes alike. The
repaid gas funds execution again, a later creation spills afresh, and
settlement is unchanged because it sums both pools. The measured
windows subtract `gas_left` readings with `SUB`, so a window repaid
more than it cost reads below zero.

A same-frame control pins the pre-existing form of that last behavior:
a local refill can already make the second of two `GAS` readings larger
than the first. Merge-time repayment extends it across a frame boundary.

Tests for [EIP-8037: State Creation Gas Cost Increase]
(https://eips.ethereum.org/EIPS/eip-8037).
"""

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    AuthorizationTuple,
    Bytecode,
    Fork,
    GasConsumer,
    Op,
    Opcode,
    StateTestFiller,
    Storage,
    Transaction,
    TransactionReceipt,
    create_op,
)

from tests.prague.eip7702_set_code_tx.spec import Spec as Spec7702

from .spec import init_code_at_high_bytes, ref_spec_8037

REFERENCE_SPEC_GIT_PATH = ref_spec_8037.git_path
REFERENCE_SPEC_VERSION = ref_spec_8037.version

SLOT_X = 1
SLOT_Y = 2
SLOT_MARKER = 3
SLOT_RESULT = 4
SLOT_INCREASED = 5
SLOT_PROBE = 6
SLOT_PROBE_RESULT = 7

# A cold set of a slot that was zero when the transaction began, and
# the warm clear that undoes it.
FRESH_SET = Op.SSTORE.with_metadata(
    key_warm=False, original_value=0, current_value=0, new_value=1
)
WARM_CLEAR = Op.SSTORE.with_metadata(
    key_warm=True, original_value=0, current_value=1, new_value=0
)


def window_cost_excess() -> Bytecode:
    """
    Return code storing the first window's cost over the second's.

    Memory holds `g0`, `g1` and `g2` at 0, 32 and 64. The stored
    value is `(g0 - g1) - (g1 - g2)`, the first window's cost minus
    the second's, computed modulo 2**256.
    """
    return FRESH_SET(
        SLOT_RESULT,
        Op.SUB(
            Op.ADD(Op.MLOAD(0), Op.MLOAD(64)),
            Op.ADD(Op.MLOAD(32), Op.MLOAD(32)),
        ),
    )


def deploy_slot_holder(pre: Alloc) -> Address:
    """
    Deploy the contract owning the slot the dispatches clear.

    It stores its calldata size, so a call carrying one byte sets the
    slot and a call carrying none clears it.
    """
    return pre.deploy_contract(code=Op.SSTORE(SLOT_X, Op.CALLDATASIZE))


def budget_above_sstore_stipend(fork: Fork, code: Bytecode) -> int:
    """
    Return a call budget leaving the child more than the stipend.

    SSTORE refuses to run with only the call stipend left, so a child
    that stores needs that much on top of what its code costs.
    """
    return fork.call_value_stipend() + 1 + code.gas_cost(fork)


def delegation_to(
    pre: Alloc, contract: Address
) -> tuple[Address, list[AuthorizationTuple]]:
    """Return a signer and the authorization delegating it to `contract`."""
    signer = pre.fund_eoa()
    return signer, [
        AuthorizationTuple(
            address=contract,
            nonce=0,
            signer=signer,
            creates_account=False,
            writes_delegation=True,
        )
    ]


def clearing_child_code(child_ending: str) -> Bytecode:
    """
    Return child code spilling a set, clearing both parent slots,
    then ending as `child_ending` says.
    """
    body = (
        FRESH_SET(SLOT_MARKER, 1)
        + WARM_CLEAR(SLOT_X, 0)
        + WARM_CLEAR(SLOT_Y, 0)
    )
    if child_ending == "stop":
        return body + Op.STOP
    if child_ending == "revert":
        return body + Op.REVERT(0, 0)
    if child_ending == "invalid":
        return body + Op.INVALID
    raise ValueError(f"unhandled child ending: {child_ending}")


def clearing_child_storage(child_ending: str) -> dict[int, int]:
    """Return the parent storage a child with this ending leaves behind."""
    if child_ending == "stop":
        return {SLOT_X: 0, SLOT_Y: 0, SLOT_MARKER: 1}
    return {SLOT_X: 1, SLOT_Y: 1, SLOT_MARKER: 0}


def clearing_probe_code(
    set_slot: Bytecode, windows: list[Bytecode]
) -> Bytecode:
    """
    Return code measuring a clearing window against a no-op window.

    `windows` holds three copies of the dispatch under test. The first
    warms the target and the slot while the slot is still zero, and
    pre-expands the measurement memory, so the two measured windows
    below are byte-identical and cost-identical. `set_slot` then sets
    the slot, spilling any state charge the reservoir cannot cover. The
    first measured window clears the slot and the second repeats as a
    no-op, so the refunded state gas is what separates their cost.
    """
    warm, first, second = windows
    return (
        warm
        + Op.MSTORE(64, 0)
        + set_slot
        + Op.MSTORE(0, Op.GAS)
        + first
        + Op.MSTORE(32, Op.GAS)
        + second
        + Op.MSTORE(64, Op.GAS)
        + Op.SSTORE(SLOT_INCREASED, Op.GT(Op.MLOAD(32), Op.MLOAD(0)))
        + window_cost_excess()
        # The marker distinguishes the pinned run from a reverted one.
        + Op.SSTORE(SLOT_MARKER, 1)
    )


@pytest.mark.parametrize(
    "fund_half_slot",
    [False, True],
    ids=["empty-reservoir", "half-slot-reservoir"],
)
@pytest.mark.valid_from("EIP8037")
def test_same_frame_refund_increases_gas_left(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    fund_half_slot: bool,
) -> None:
    """
    Test a same-frame refill can increase `gas_left` between reads.

    This is the local control for the cross-frame behavior: a fresh set
    spills from `gas_left`, the first measured window clears it in the
    same frame, and the second repeats the clear as a no-op. The two
    windows have identical bytecode and execution cost, so their exact
    difference is the repaid spill. With half a slot funded by the
    reservoir, only half of the full-slot refund repays the spill.
    """
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)
    reservoir = sstore_state_gas // 2 if fund_half_slot else 0

    window = Op.SSTORE(SLOT_X, 0)
    contract = pre.deploy_contract(
        code=clearing_probe_code(FRESH_SET(SLOT_X, 1), [window] * 3)
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=reservoir,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                SLOT_X: 0,
                SLOT_MARKER: 1,
                SLOT_INCREASED: 0,
                SLOT_RESULT: 0,
            }
        )
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize("call_opcode", [Op.CALLCODE, Op.DELEGATECALL])
@pytest.mark.parametrize(
    "fund_half_slot",
    [False, True],
    ids=["empty-reservoir", "half-slot-reservoir"],
)
@pytest.mark.valid_from("EIP8037")
def test_cross_frame_refund_repays_spill_at_merge(
    state_test: StateTestFiller,
    pre: Alloc,
    call_opcode: Opcode,
    fork: Fork,
    fund_half_slot: bool,
) -> None:
    """
    Test a cross-frame slot-clear refill does not repay spill at merge.

    The frame sets a slot with an empty or half-slot reservoir, spilling
    the uncovered state charge from `gas_left`. A child sharing the
    caller's storage clears the slot, holding the refill back until
    settlement rather than crediting `gas_left` at the merge.
    """
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)
    reservoir = sstore_state_gas // 2 if fund_half_slot else 0

    clearer = pre.deploy_contract(code=Op.SSTORE(SLOT_X, 0))
    window = Op.POP(call_opcode(address=clearer))
    contract = pre.deploy_contract(
        code=clearing_probe_code(Op.SSTORE(SLOT_X, 1), [window] * 3)
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=reservoir,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                SLOT_X: 0,
                SLOT_MARKER: 1,
                SLOT_INCREASED: 0,
                SLOT_RESULT: 0,
            }
        )
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize(
    "fund_half_slot",
    [False, True],
    ids=["empty-reservoir", "half-slot-reservoir"],
)
@pytest.mark.valid_from("EIP8037")
def test_cross_frame_refund_repays_spill_in_inner_frame(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    fund_half_slot: bool,
) -> None:
    """
    Test a slot-clear refill does not repay spill at an inner frame's merge.
    """
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)
    reservoir = sstore_state_gas // 2 if fund_half_slot else 0

    clearer = pre.deploy_contract(code=Op.SSTORE(SLOT_X, 0))
    window = Op.POP(Op.DELEGATECALL(address=clearer))
    middle = pre.deploy_contract(
        code=clearing_probe_code(Op.SSTORE(SLOT_X, 1), [window] * 3)
    )
    contract = pre.deploy_contract(
        code=Op.POP(Op.DELEGATECALL(address=middle))
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=reservoir,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                SLOT_X: 0,
                SLOT_MARKER: 1,
                SLOT_INCREASED: 0,
                SLOT_RESULT: 0,
            }
        )
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_cross_frame_credit_returns_at_settlement(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test the cross-frame credit refunds the sender at settlement.

    The frame's spilled set is cleared by a child and the merge repays
    the credit. Settlement sums `gas_left` and the reservoir, so the
    spilled charge and the credit cancel wherever the credit sits and
    the receipt carries no state term at all.
    """
    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()()

    clearer_code = WARM_CLEAR(SLOT_X, 0)
    clearer = pre.deploy_contract(code=clearer_code)

    # A budget covering the child's SSTORE stipend sentry through the
    # clear, so the child succeeds and returns the sentry unspent.
    child_budget = budget_above_sstore_stipend(fork, clearer_code)
    code = Op.SSTORE(
        SLOT_X,
        1,
        key_warm=False,
        original_value=0,
        current_value=0,
        new_value=1,
    ) + Op.POP(
        Op.DELEGATECALL(gas=child_budget, address=clearer, address_warm=False)
    )
    contract = pre.deploy_contract(code=code)

    before_refund = (
        intrinsic_cost + code.gas_cost(fork) + clearer_code.gas_cost(fork)
    )
    # Clearing the slot back to its original value refunds both the
    # state cost and the write cost through the capped refund counter.
    restore_refund = clearer_code.refund(fork)
    expected_gas_used = before_refund - min(
        before_refund // fork.max_refund_quotient(), restore_refund
    )
    # The post-refund usage must clear the calldata floor, or the
    # floor masks a lost or doubled credit.
    assert expected_gas_used > fork.transaction_data_floor_cost_calculator()(
        data=b""
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=expected_gas_used
        ),
    )

    post = {contract: Account(storage={SLOT_X: 0})}
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_cross_frame_credit_funds_state_at_full_price(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test the sender pays full price for state after a cross-frame refund.

    A fresh set spills, a delegated child clears it, and the clear
    refunds through the capped refund counter. A second fresh set then
    survives and is billed at the full state price, so routing a refund
    through another frame buys no discount on state that persists. What
    the later set costs `gas_left` is measured in
    `test_later_set_spills_after_repayment`.
    """
    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()()

    clearer_code = WARM_CLEAR(SLOT_X, 0)
    clearer = pre.deploy_contract(code=clearer_code)
    child_budget = budget_above_sstore_stipend(fork, clearer_code)

    code = (
        FRESH_SET(SLOT_X, 1)
        + Op.POP(
            Op.DELEGATECALL(
                gas=child_budget, address=clearer, address_warm=False
            )
        )
        + FRESH_SET(SLOT_Y, 1)
    )
    contract = pre.deploy_contract(code=code)

    before_refund = (
        intrinsic_cost + code.gas_cost(fork) + clearer_code.gas_cost(fork)
    )
    restore_refund = clearer_code.refund(fork)
    expected_gas_used = before_refund - min(
        before_refund // fork.max_refund_quotient(), restore_refund
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=expected_gas_used
        ),
    )

    post = {contract: Account(storage={SLOT_X: 0, SLOT_Y: 1})}
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_later_set_spills_after_repayment(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test a later creation spills again once the credit is repaid.

    After the cross-frame clear, the merge repays the credit into
    `gas_left`, so a fresh set finds the reservoir empty and spills:
    `gas_left` drops by the execution premium plus the slot's state
    cost across the set window. What the sender pays for it is checked
    in `test_cross_frame_credit_funds_state_at_full_price`.
    """
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)

    clearer_code = WARM_CLEAR(SLOT_X, 0)
    clearer = pre.deploy_contract(code=clearer_code)
    child_budget = budget_above_sstore_stipend(fork, clearer_code)

    window_1 = FRESH_SET(SLOT_Y, 1)
    window_2 = Op.SSTORE.with_metadata(
        key_warm=True,
        original_value=0,
        current_value=1,
        new_value=1,
    )(SLOT_Y, 1)
    # The windows are byte-identical, so the excess is the fresh set's
    # execution premium plus whatever its state charge takes from
    # `gas_left`. The merge drained the credit into `gas_left`, so the
    # set spills and the excess is the premium plus the state cost.
    execution_premium = window_1.execution_cost(
        fork
    ) - window_2.execution_cost(fork)

    code = (
        Op.MSTORE(64, 0, new_memory_size=96, old_memory_size=0)
        + FRESH_SET(SLOT_X, 1)
        + Op.POP(
            Op.DELEGATECALL(
                gas=child_budget, address=clearer, address_warm=False
            )
        )
        + Op.MSTORE(0, Op.GAS)
        + window_1
        + Op.MSTORE(32, Op.GAS)
        + window_2
        + Op.MSTORE(64, Op.GAS)
        + window_cost_excess()
    )
    contract = pre.deploy_contract(code=code)

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                SLOT_X: 0,
                SLOT_Y: 1,
                SLOT_RESULT: execution_premium + sstore_state_gas,
            }
        )
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_repaid_credit_funds_execution(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test the repaid credit funds execution work.

    The gas limit covers the transaction only up to the clearing
    child's merge plus a sliver. An execution tail worth more than the
    sliver but no more than the repaid credit follows and completes:
    the merge repaid the spill into `gas_left`, where the tail spends
    it, and the receipt bills the tail against the repayment.
    """
    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()()
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)

    clearer_code = WARM_CLEAR(SLOT_X, 0)
    clearer = pre.deploy_contract(code=clearer_code)

    head = (
        FRESH_SET(SLOT_MARKER, 1)
        + FRESH_SET(SLOT_X, 1)
        + Op.POP(
            Op.DELEGATECALL(gas=Op.GAS, address=clearer, address_warm=False)
        )
    )
    # A sliver covering the child's SSTORE stipend sentry through the
    # one-in-64 withholding. It survives the merge unspent.
    sliver = budget_above_sstore_stipend(fork, clearer_code) * 64 // 63 + 1
    # Burn exactly the repaid state-gas credit as execution gas. The
    # tail must overrun the sliver yet fit inside that credit, or the
    # completion stops demonstrating the repayment buys execution.
    tail_cost = sstore_state_gas
    assert sliver < tail_cost
    tail = GasConsumer(gas=tail_cost, fork=fork)
    code = head + tail
    contract = pre.deploy_contract(code=code)

    gas_limit = (
        intrinsic_cost
        + head.gas_cost(fork)
        + clearer_code.gas_cost(fork)
        + sliver
    )

    # The clearing child's refill is held back until settlement and does
    # not enter `gas_left` at the merge, so the execution tail overruns
    # the sliver and OOGs the top frame.
    expected_gas_used = gas_limit

    tx = Transaction(
        to=contract,
        gas_limit=gas_limit,
        sender=pre.fund_eoa(),
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=expected_gas_used
        ),
    )

    post = {contract: Account(storage={})}
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize(
    "raises_credit",
    [
        pytest.param(True, id="credit_repaid_probe_oogs"),
        pytest.param(False, id="no_credit_probe_oogs"),
    ],
)
@pytest.mark.parametrize("depth", ["sibling", "grandchild"])
@pytest.mark.valid_from("EIP8037")
def test_call_frame_credit_repays_spill_before_fixed_gas_probe(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    depth: str,
    raises_credit: bool,
) -> None:
    """
    Test a plain CALL credit repays the spill before a fixed-gas probe.

    The transaction starts with no reservoir. The spilling frame merges
    its claim into the parent. A sibling or grandchild clears the slot,
    and its credit repays that claim when it reaches the parent. The
    probe receives only its execution cost, so it cannot fund its state
    charge from the repaid gas. It fails with or without the clear.
    """
    # A CALL callee owns its storage, so splitting the charge from the
    # credit needs one contract entered twice rather than two contracts.
    toggler = pre.deploy_contract(
        code=Op.SSTORE(SLOT_X, Op.ISZERO(Op.SLOAD(SLOT_X)))
    )
    clearing_target = toggler
    if depth == "grandchild":
        clearing_target = pre.deploy_contract(
            code=Op.POP(Op.CALL(gas=Op.GAS, address=toggler))
        )

    # Without the second entry no credit is raised. Both cases leave
    # the reservoir empty when the fixed-gas probe runs.
    if not raises_credit:
        clearing_target = pre.deploy_contract(code=Op.STOP)

    probe_storage = Storage()
    probe_code = Op.SSTORE(probe_storage.store_next(0, "probe_ran"), 1)
    probe = pre.deploy_contract(probe_code)
    probe_stipend = probe_code.execution_cost(fork)

    entry = pre.deploy_contract(
        code=(
            Op.POP(Op.CALL(gas=Op.GAS, address=toggler))
            + Op.POP(Op.CALL(gas=Op.GAS, address=clearing_target))
            + Op.POP(Op.CALL(gas=probe_stipend, address=probe))
        )
    )

    tx = Transaction(
        to=entry,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        toggler: Account(storage={SLOT_X: 0 if raises_credit else 1}),
        probe: Account(storage=probe_storage),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize(
    "clearing_frame_ending",
    [
        pytest.param("stop", id="child_succeeds"),
        pytest.param("revert", id="child_reverts"),
    ],
)
@pytest.mark.valid_from("EIP8037")
def test_fixed_gas_probe_after_clearing_frame(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    clearing_frame_ending: str,
) -> None:
    """
    Test a fixed-gas probe after a successful or reverted clear.

    A successful child repays the caller's spill, leaving no reservoir
    for the probe. A reverted child rolls back both the clear and its
    credit. The probe cannot pay its state charge in either case; the
    caller's slot distinguishes the committed clear from the rollback.
    """
    credit_survives = clearing_frame_ending == "stop"

    clear = Op.SSTORE.with_metadata(
        key_warm=True, original_value=0, current_value=1, new_value=0
    )(SLOT_X, 0)
    clearer_code = clear + (Op.STOP if credit_survives else Op.REVERT(0, 0))
    clearer = pre.deploy_contract(code=clearer_code)

    probe_storage = Storage()
    probe_code = Op.SSTORE(probe_storage.store_next(0, "probe_ran"), 1)
    probe = pre.deploy_contract(probe_code)
    probe_stipend = probe_code.execution_cost(fork)

    entry = pre.deploy_contract(
        code=(
            Op.SSTORE(SLOT_X, 1)
            + Op.POP(Op.DELEGATECALL(gas=Op.GAS, address=clearer))
            + Op.POP(Op.CALL(gas=probe_stipend, address=probe))
        )
    )

    tx = Transaction(
        to=entry,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        entry: Account(storage={SLOT_X: 0 if credit_survives else 1}),
        probe: Account(storage=probe_storage),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_repaid_credit_enters_next_call_forwarding_base(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test deferred slot-clear refill does not enter the next call's 63/64 base.

    Without mid-transaction repayment, the parent has one gas less than
    the smallest base whose 63/64 allowance can run the probe. The
    clearing child's refill is held back until settlement, so the probe
    call's forwarding base stays 1 gas short and the probe OOGs.
    """
    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()()
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)

    clearer_code = WARM_CLEAR(SLOT_X, 0)
    clearer = pre.deploy_contract(code=clearer_code)
    head = FRESH_SET(SLOT_X, 1) + Op.POP(
        Op.DELEGATECALL(gas=Op.GAS, address=clearer, address_warm=False)
    )

    # A nonzero-to-nonzero write leaves an observable success marker
    # without consuming state gas of its own.
    probe_code = Op.SSTORE.with_metadata(
        key_warm=False,
        original_value=1,
        current_value=1,
        new_value=2,
    )(SLOT_X, 2)
    probe = pre.deploy_contract(code=probe_code, storage={SLOT_X: 1})
    probe_execution = probe_code.execution_cost(fork)

    # Find the smallest base whose EIP-150 allowance
    # `base - base // 64` covers the probe exactly.
    forwarding_base = (probe_execution - 1) * 64 // 63 + 1
    assert (
        forwarding_base - 1 - (forwarding_base - 1) // 64
        < probe_execution
        <= forwarding_base - forwarding_base // 64
    )
    repaid_base = forwarding_base - 1 + sstore_state_gas
    assert repaid_base - repaid_base // 64 >= probe_execution

    probe_call = Op.CALL(
        gas=0xFFFFFFFF,
        address=probe,
        value=0,
        args_offset=0,
        args_size=0,
        ret_offset=0,
        ret_size=0,
        address_warm=False,
    )
    # With no mid-transaction repayment, paying the call's own execution
    # cost leaves `forwarding_base - 1`, so the probe OOGs by construction.
    sliver = probe_call.execution_cost(fork) + forwarding_base - 1
    clear_budget = budget_above_sstore_stipend(fork, clearer_code)
    assert sliver >= clear_budget * 64 // 63 + 1
    gas_limit = (
        intrinsic_cost
        + head.gas_cost(fork)
        + clearer_code.gas_cost(fork)
        + sliver
    )

    contract = pre.deploy_contract(code=head + Op.POP(probe_call))
    tx = Transaction(
        to=contract,
        gas_limit=gas_limit,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(storage={SLOT_X: 0}),
        probe: Account(storage={SLOT_X: 1}),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize("child_ending", ["stop", "revert", "invalid"])
@pytest.mark.valid_from("EIP8037")
def test_child_clear_repays_own_spill_first(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    child_ending: str,
) -> None:
    """
    Test the cross-slot LIFO split of a cross-frame refund in a child.

    The parent spills two fresh sets; a delegated child spills a set
    of its own, then clears both parent slots. The first credit repays
    the child's borrow, the second parks in the reservoir and repays
    one parent spill at the merge, and a failing child discards the
    parked credit with its rollback. What the call costs `gas_left` is
    measured in `test_child_clear_window_cost`.
    """
    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()()
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)

    child_code = clearing_child_code(child_ending)
    child = pre.deploy_contract(code=child_code)
    child_budget = budget_above_sstore_stipend(fork, child_code)

    code = (
        FRESH_SET(SLOT_X, 1)
        + FRESH_SET(SLOT_Y, 1)
        + Op.POP(
            Op.DELEGATECALL(
                gas=child_budget, address=child, address_warm=False
            )
        )
    )
    contract = pre.deploy_contract(code=code)

    parent_exec = code.execution_cost(fork)
    if child_ending == "stop":
        before_refund = (
            intrinsic_cost + code.gas_cost(fork) + child_code.gas_cost(fork)
        )
        restore_refund = child_code.refund(fork)
        expected_gas_used = before_refund - min(
            before_refund // fork.max_refund_quotient(), restore_refund
        )
    elif child_ending == "revert":
        expected_gas_used = (
            intrinsic_cost
            + parent_exec
            + child_code.execution_cost(fork)
            + 2 * sstore_state_gas
        )
    else:
        expected_gas_used = (
            intrinsic_cost + parent_exec + child_budget + 2 * sstore_state_gas
        )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=expected_gas_used
        ),
    )

    post = {contract: Account(storage=clearing_child_storage(child_ending))}
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize("child_ending", ["stop", "revert", "invalid"])
@pytest.mark.valid_from("EIP8037")
def test_child_clear_window_cost(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    child_ending: str,
) -> None:
    """
    Test a clearing child's `gas_left` window cost across endings.

    The same shape as `test_child_clear_repays_own_spill_first`, with
    the call bracketed by two `GAS` reads. Because slot-clear refills
    are held back until settlement, a successful child neither repays
    its own spilled set nor repays a parent spill at the merge. A
    reverting child restores its own spilled set on rollback, and an
    invalid child burns its whole budget.
    """
    child_code = clearing_child_code(child_ending)
    child = pre.deploy_contract(code=child_code)
    child_budget = budget_above_sstore_stipend(fork, child_code)

    call_window = Op.POP(
        Op.DELEGATECALL(gas=child_budget, address=child, address_warm=False)
    )
    code = (
        FRESH_SET(SLOT_X, 1)
        + FRESH_SET(SLOT_Y, 1)
        + Op.MSTORE(32, 0, new_memory_size=64, old_memory_size=0)
        + Op.MSTORE(0, Op.GAS)
        + call_window
        + Op.MSTORE(32, Op.GAS)
        + FRESH_SET(SLOT_RESULT, Op.SUB(Op.MLOAD(0), Op.MLOAD(32)))
    )
    contract = pre.deploy_contract(code=code)

    # An invalid child burns its whole budget; a reverting child rolls
    # back its spilled set; a successful child keeps its spilled set
    # charged because the slot-clear refills are held back until
    # settlement.
    if child_ending == "invalid":
        child_consumed = child_budget
    elif child_ending == "revert":
        child_consumed = child_code.execution_cost(fork)
    else:
        child_consumed = child_code.gas_cost(fork)
    # Gas measured between the two reads: the first stamp's store, the
    # call window, the child's consumption, and the second read itself.
    window_cost = (
        Op.MSTORE(0, Op.GAS).gas_cost(fork)
        + call_window.execution_cost(fork)
        + child_consumed
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                **clearing_child_storage(child_ending),
                SLOT_RESULT: window_cost,
            }
        )
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_cross_frame_refund_after_delegation_spill(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test a cross-frame refund after the sender's delegation spilled.

    A set-code transaction with an empty reservoir pays its delegation
    from `gas_left` and commits that spill before the code runs. The
    code spills a fresh set and a delegated child clears it. The merge
    repays the code's spill but not the committed delegation spill, so
    the delegation stays billed. What the call costs `gas_left` is
    measured in `test_delegation_spill_window_cost`.
    """
    child_code = WARM_CLEAR(SLOT_X, 0)
    child = pre.deploy_contract(code=child_code)
    child_budget = budget_above_sstore_stipend(fork, child_code)

    code = FRESH_SET(SLOT_X, 1) + Op.POP(
        Op.DELEGATECALL(gas=child_budget, address=child, address_warm=False)
    )
    contract = pre.deploy_contract(code=code)

    signer, authorization_list = delegation_to(pre, contract)

    gas_used = (
        fork.transaction_intrinsic_cost_calculator()(
            authorization_list_or_count=authorization_list,
            return_cost_deducted_prior_execution=True,
        )
        + fork.transaction_top_frame_execution_gas(
            authorizations=authorization_list
        )
        + fork.transaction_top_frame_state_gas(
            authorizations=authorization_list
        )
        + code.gas_cost(fork)
        + child_code.gas_cost(fork)
    )
    refund = child_code.refund(fork)
    gas_used -= min(gas_used // fork.max_refund_quotient(), refund)

    tx = Transaction(
        to=contract,
        authorization_list=authorization_list,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
        expected_receipt=TransactionReceipt(cumulative_gas_used=gas_used),
    )

    post = {
        contract: Account(storage={SLOT_X: 0}),
        signer: Account(code=Spec7702.delegation_designation(contract)),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.valid_from("EIP8037")
def test_delegation_spill_window_cost(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Test the clearing call does not repay `gas_left` at the merge.

    The same shape as `test_cross_frame_refund_after_delegation_spill`,
    with the call bracketed by two `GAS` reads. Because the slot-clear
    refill is held back until settlement, the window costs the call and
    child's execution alone.
    """
    child_code = WARM_CLEAR(SLOT_X, 0)
    child = pre.deploy_contract(code=child_code)
    child_budget = budget_above_sstore_stipend(fork, child_code)

    call = Op.POP(
        Op.DELEGATECALL(gas=child_budget, address=child, address_warm=False)
    )
    code = (
        Op.MSTORE(32, 0, new_memory_size=64, old_memory_size=0)
        + FRESH_SET(SLOT_X, 1)
        + Op.MSTORE(0, Op.GAS)
        + call
        + Op.MSTORE(32, Op.GAS)
        + FRESH_SET(SLOT_RESULT, Op.SUB(Op.MLOAD(0), Op.MLOAD(32)))
    )
    contract = pre.deploy_contract(code=code)

    signer, authorization_list = delegation_to(pre, contract)

    # The window runs from one GAS read to the next: the store of the
    # first read, the call and the second read.
    call_cost = (
        Op.MSTORE(0, Op.GAS).gas_cost(fork)
        + call.execution_cost(fork)
        + child_code.execution_cost(fork)
    )

    tx = Transaction(
        to=contract,
        authorization_list=authorization_list,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                SLOT_X: 0,
                SLOT_RESULT: call_cost,
            }
        ),
        signer: Account(code=Spec7702.delegation_designation(contract)),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize("reservoir_slots", [0, 1, 2])
@pytest.mark.valid_from("EIP8037")
def test_cross_frame_refund_with_reservoir_grant(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    reservoir_slots: int,
) -> None:
    """
    Test the receipt is the same however much reservoir was bought.

    The reservoir covers none, one or both of the parent's two sets
    and the rest spill. A delegated child clears both slots and a
    later set draws on whatever is left. The receipt is the same in
    every case, so buying reservoir up front costs the sender nothing
    and saves nothing. What the call and the set cost `gas_left` is
    measured in `test_reservoir_grant_window_costs`.
    """
    child_code = WARM_CLEAR(SLOT_X, 0) + WARM_CLEAR(SLOT_Y, 0)
    child = pre.deploy_contract(code=child_code)
    child_budget = budget_above_sstore_stipend(fork, child_code)

    code = (
        FRESH_SET(SLOT_X, 1)
        + FRESH_SET(SLOT_Y, 1)
        + Op.POP(
            Op.DELEGATECALL(
                gas=child_budget, address=child, address_warm=False
            )
        )
        + FRESH_SET(SLOT_PROBE, 1)
    )
    contract = pre.deploy_contract(code=code)

    gas_used = (
        fork.transaction_intrinsic_cost_calculator()()
        + code.gas_cost(fork)
        + child_code.gas_cost(fork)
    )
    refund = child_code.refund(fork)
    gas_used -= min(gas_used // fork.max_refund_quotient(), refund)

    tx = Transaction(
        to=contract,
        state_gas_reservoir=(
            reservoir_slots * Op.SSTORE(new_value=1).state_cost(fork)
        ),
        sender=pre.fund_eoa(),
        expected_receipt=TransactionReceipt(cumulative_gas_used=gas_used),
    )

    post = {contract: Account(storage={SLOT_X: 0, SLOT_Y: 0, SLOT_PROBE: 1})}
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.parametrize("reservoir_slots", [0, 1, 2])
@pytest.mark.valid_from("EIP8037")
def test_reservoir_grant_window_costs(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    reservoir_slots: int,
) -> None:
    """
    Test `gas_left` window costs across reservoir grants when clears defer.

    The same shape as `test_cross_frame_refund_with_reservoir_grant`,
    with the call and a later set each bracketed by `GAS` reads. The
    parent's two sets consume any reservoir up to two slots, and the
    child's two clears hold their refills back until settlement, so the
    call window never shrinks `gas_left` and the later probe set always
    spills from `gas_left`.
    """
    sstore_state_gas = Op.SSTORE(new_value=1).state_cost(fork)

    child_code = WARM_CLEAR(SLOT_X, 0) + WARM_CLEAR(SLOT_Y, 0)
    child = pre.deploy_contract(code=child_code)
    child_budget = budget_above_sstore_stipend(fork, child_code)

    call = Op.POP(
        Op.DELEGATECALL(gas=child_budget, address=child, address_warm=False)
    )
    probe = FRESH_SET(SLOT_PROBE, 1)
    code = (
        Op.MSTORE(64, 0, new_memory_size=96, old_memory_size=0)
        + FRESH_SET(SLOT_X, 1)
        + FRESH_SET(SLOT_Y, 1)
        + Op.MSTORE(0, Op.GAS)
        + call
        + Op.MSTORE(32, Op.GAS)
        + probe
        + Op.MSTORE(64, Op.GAS)
        + FRESH_SET(SLOT_RESULT, Op.SUB(Op.MLOAD(0), Op.MLOAD(32)))
        + FRESH_SET(SLOT_PROBE_RESULT, Op.SUB(Op.MLOAD(32), Op.MLOAD(64)))
    )
    contract = pre.deploy_contract(code=code)

    # A window runs from one GAS read to the next: the store of the
    # first read, the window's code and the second read.
    stamp_cost = Op.MSTORE(0, Op.GAS).gas_cost(fork)
    call_cost = (
        stamp_cost
        + call.execution_cost(fork)
        + child_code.execution_cost(fork)
    )
    probe_cost = stamp_cost + probe.execution_cost(fork)

    tx = Transaction(
        to=contract,
        state_gas_reservoir=reservoir_slots * sstore_state_gas,
        sender=pre.fund_eoa(),
    )

    post = {
        contract: Account(
            storage={
                SLOT_X: 0,
                SLOT_Y: 0,
                SLOT_PROBE: 1,
                SLOT_RESULT: call_cost,
                SLOT_PROBE_RESULT: probe_cost + sstore_state_gas,
            }
        )
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.with_all_call_opcodes(
    # A static child cannot write, so it can never refund.
    selector=lambda call_opcode: call_opcode != Op.STATICCALL
)
@pytest.mark.valid_from("EIP8037")
def test_cross_frame_refund_repays_spill_at_a_call(
    state_test: StateTestFiller,
    pre: Alloc,
    call_opcode: Opcode,
) -> None:
    """
    Test a call merging a slot-clear refill does not repay the spill.

    A holder contract owns the cleared slot, so every dispatch reaches
    it the same way. Because the refill is held back until settlement,
    the clearing window costs the same `gas_left` as the no-op window.
    """
    holder = deploy_slot_holder(pre)
    set_slot = Op.POP(Op.CALL(address=holder, args_size=1))
    clearer = pre.deploy_contract(code=Op.CALL(address=holder))
    window = Op.POP(call_opcode(address=clearer))
    contract = pre.deploy_contract(
        code=clearing_probe_code(set_slot, [window] * 3)
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        holder: Account(storage={SLOT_X: 0}),
        contract: Account(
            storage={
                SLOT_MARKER: 1,
                SLOT_INCREASED: 0,
                SLOT_RESULT: 0,
            }
        ),
    }
    state_test(pre=pre, post=post, tx=tx)


@pytest.mark.with_all_create_opcodes
@pytest.mark.valid_from("EIP8037")
def test_cross_frame_refund_repays_spill_at_a_create(
    state_test: StateTestFiller,
    pre: Alloc,
    create_opcode: Opcode,
) -> None:
    """
    Test a create merging a slot-clear refill does not repay the spill.

    The initcode reaches a holder contract that clears its own slot.
    Because the slot-clear refill is held back until settlement, the
    clearing window costs the same `gas_left` as the no-op window.
    """
    holder = deploy_slot_holder(pre)
    set_slot = Op.POP(Op.CALL(address=holder, args_size=1))
    # Pushing the zero arguments with PUSH0 keeps the initcode inside a
    # single memory word.
    initcode = Op.CALL(
        Op.GAS, holder, Op.PUSH0, Op.PUSH0, Op.PUSH0, Op.PUSH0, Op.PUSH0
    )
    mstore_value, size = init_code_at_high_bytes(initcode)

    # The probe measures with memory below 96, so the initcode sits
    # above it.
    code_offset = 96

    def window(salt: int) -> Bytecode:
        # A repeated CREATE2 salt would collide with the account the
        # previous window created.
        return Op.POP(
            create_op(create_opcode, offset=code_offset, size=size, salt=salt)
        )

    contract = pre.deploy_contract(
        code=Op.MSTORE(code_offset, mstore_value)
        + clearing_probe_code(set_slot, [window(salt) for salt in range(3)])
    )

    tx = Transaction(
        to=contract,
        state_gas_reservoir=0,
        sender=pre.fund_eoa(),
    )

    post = {
        holder: Account(storage={SLOT_X: 0}),
        contract: Account(
            storage={
                SLOT_MARKER: 1,
                SLOT_INCREASED: 0,
                SLOT_RESULT: 0,
            }
        ),
    }
    state_test(pre=pre, post=post, tx=tx)
