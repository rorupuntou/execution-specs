"""Witness state collection scenarios for state writes."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Bytes,
    ExecutionWitnessStateExpectation,
    Fork,
    Op,
    Transaction,
)
from execution_testing.forks import Amsterdam

from .gas_helpers import empty_account_value_transfer_gas_limit
from .spec import ref_spec_8025
from .state_helpers import (
    as_storage,
    build_large_storage,
    collect_account_post_state_only_nodes,
    collect_account_proof_nodes,
    collect_storage_post_state_only_nodes,
    collect_storage_proof_nodes,
    find_account_with_shared_secured_nibble,
    large_storage_value,
    merge_with_amsterdam_pre_alloc,
)

pytestmark = pytest.mark.valid_at("Amsterdam")

REFERENCE_SPEC_GIT_PATH = ref_spec_8025.git_path
REFERENCE_SPEC_VERSION = ref_spec_8025.version


@pytest.mark.parametrize(
    "revert",
    [
        pytest.param(False, id="stop"),
        pytest.param(True, id="revert"),
    ],
)
def test_witness_state_sstore_without_explicit_read_contains_storage_proof(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    revert: bool,
) -> None:
    """
    Dirty storage writes should still include the pre-state proof.

    A reverted SSTORE should still leave its proof nodes in witness state.
    """
    storage = build_large_storage([1])
    proof_nodes = collect_storage_proof_nodes(storage, [1])
    assert proof_nodes

    new_value = large_storage_value(9)
    terminator = Op.REVERT(0, 0) if revert else Op.STOP
    contract = pre.deploy_contract(
        code=Op.SSTORE(1, new_value) + terminator,
        storage=as_storage(storage),
    )
    sender = pre.fund_eoa()
    tx = Transaction(sender=sender, to=contract, gas_limit=500_000)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage=storage if revert else {1: new_value}),
        },
    )


def test_witness_state_sstore_new_slot_omits_post_state_nodes(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Inserting a new slot should include the pre-state absence proof only.

    Nodes created solely by the post-state insertion must not leak into the
    witness.
    """
    pre_storage = build_large_storage([1, 2])
    insert_slot = 3
    insert_value = large_storage_value(insert_slot)
    post_storage = {
        **pre_storage,
        insert_slot: insert_value,
    }
    proof_nodes = collect_storage_proof_nodes(pre_storage, [insert_slot])
    post_state_only_nodes = collect_storage_post_state_only_nodes(
        pre_storage=pre_storage,
        post_storage=post_storage,
        slot=insert_slot,
        pre_state_reference_slots=[insert_slot],
    )
    assert proof_nodes
    assert post_state_only_nodes

    contract = pre.deploy_contract(
        code=Op.SSTORE(insert_slot, insert_value) + Op.STOP,
        storage=as_storage(pre_storage),
    )
    sender = pre.fund_eoa()
    tx = Transaction(sender=sender, to=contract, gas_limit=500_000)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes,
                        nodes_absent=post_state_only_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage=post_storage),
        },
    )


def test_witness_state_sstore_into_empty_storage_omits_post_state_nodes(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Empty pre-state storage should not require any storage proof nodes.

    The empty-trie RLP sentinel and nodes created solely by the insertion
    are not pre-state material and must not appear in the witness.
    """
    insert_slot = 1
    insert_value = large_storage_value(insert_slot)
    pre_storage: dict[int, int] = {}
    post_storage = {insert_slot: insert_value}
    proof_nodes = collect_storage_proof_nodes(pre_storage, [insert_slot])
    post_state_only_nodes = collect_storage_post_state_only_nodes(
        pre_storage=pre_storage,
        post_storage=post_storage,
        slot=insert_slot,
        pre_state_reference_slots=[insert_slot],
    )
    assert not proof_nodes
    assert post_state_only_nodes
    empty_trie_sentinel = Bytes(b"\x80")

    contract = pre.deploy_contract(
        code=Op.SSTORE(insert_slot, insert_value) + Op.STOP,
        storage=as_storage(pre_storage),
    )
    sender = pre.fund_eoa()
    tx = Transaction(sender=sender, to=contract, gas_limit=500_000)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_absent=post_state_only_nodes
                        + [empty_trie_sentinel],
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            contract: Account(storage=post_storage),
        },
    )


def test_witness_state_new_account_omits_post_state_nodes(
    fork: Fork,
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Creating a new account should include the pre-state absence proof only.

    Nodes created solely by the post-state insertion of the new account
    must not leak into the witness.
    """
    sender = pre.fund_eoa()
    recipient = pre.fund_eoa(amount=0)
    # Add one untouched sibling under the recipient's secured-trie prefix so
    # the absence proof extends below the shared root node.
    sibling = find_account_with_shared_secured_nibble(
        recipient,
        {
            sender,
            recipient,
            *Amsterdam.execution_witness_implicit_code_addresses(),
        },
    )
    pre.fund_address(sibling, 1)

    full_alloc = merge_with_amsterdam_pre_alloc(pre)
    proof_nodes = collect_account_proof_nodes(full_alloc, [recipient])
    assert proof_nodes

    tx = Transaction(
        sender=sender,
        to=recipient,
        value=1,
        gas_limit=empty_account_value_transfer_gas_limit(fork),
    )

    post_alloc = Alloc.merge(
        full_alloc,
        Alloc(
            {
                sender: Account(nonce=1),
                recipient: Account(balance=1),
            }
        ),
    )
    post_state_only_nodes = collect_account_post_state_only_nodes(
        full_alloc,
        post_alloc,
        recipient,
        [recipient, sender, sibling],
    )
    assert post_state_only_nodes

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_execution_witness_state=(
                    ExecutionWitnessStateExpectation(
                        nodes_present=proof_nodes,
                        nodes_absent=post_state_only_nodes,
                    )
                ),
            )
        ],
        post={
            sender: Account(nonce=1),
            recipient: Account(balance=1),
            sibling: Account(balance=1),
        },
    )
