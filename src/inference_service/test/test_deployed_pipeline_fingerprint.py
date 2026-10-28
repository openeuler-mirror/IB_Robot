"""Tests for composed distributed deployment identity."""

from inference_service.distributed.types import deployed_pipeline_fingerprint


def test_contract_changes_deployed_pipeline_fingerprint():
    ownership = {"preprocessor": "cloud", "postprocessor": "cloud"}

    first = deployed_pipeline_fingerprint(
        "manifest", "contract-a", execution_mode="distributed", processor_ownership=ownership
    )
    second = deployed_pipeline_fingerprint(
        "manifest", "contract-b", execution_mode="distributed", processor_ownership=ownership
    )

    assert first != second


def test_processor_ownership_order_does_not_change_fingerprint():
    first = deployed_pipeline_fingerprint(
        "manifest",
        "contract",
        execution_mode="distributed",
        processor_ownership={"image": "cloud", "state": "edge"},
    )
    second = deployed_pipeline_fingerprint(
        "manifest",
        "contract",
        execution_mode="distributed",
        processor_ownership={"state": "edge", "image": "cloud"},
    )

    assert first == second


def test_distributed_cloud_processor_boundary_is_part_of_fingerprint():
    cloud_owned = {"preprocessor": "cloud", "postprocessor": "cloud"}
    edge_owned = {"preprocessor": "edge", "postprocessor": "edge"}

    assert deployed_pipeline_fingerprint(
        "manifest", "contract", execution_mode="distributed", processor_ownership=cloud_owned
    ) != deployed_pipeline_fingerprint(
        "manifest", "contract", execution_mode="distributed", processor_ownership=edge_owned
    )


def test_execution_mode_is_part_of_composed_fingerprint():
    ownership = {"preprocessor": "cloud", "postprocessor": "cloud"}

    assert deployed_pipeline_fingerprint(
        "manifest", "contract", execution_mode="distributed", processor_ownership=ownership
    ) != deployed_pipeline_fingerprint(
        "manifest", "contract", execution_mode="monolithic", processor_ownership=ownership
    )
