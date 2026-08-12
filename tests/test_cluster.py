from __future__ import annotations

import pytest

from yeetllm.cluster import ClusterCoordinator, detect_cluster, validate_cluster_ports
from yeetllm.config import ClusterConfig, RuntimeValidationError


def cluster_env() -> dict[str, str]:
    return {
        "PRIMARY_ADDR": "10.0.0.1",
        "MASTER_ADDR": "10.0.0.1",
        "PRIMARY_PORT": "29500",
        "MASTER_PORT": "29500",
        "NODE_ADDR": "10.0.0.2",
        "NODE_RANK": "1",
        "NUM_NODES": "2",
        "NUM_TRAINERS": "8",
        "HOST_NODE_ADDR": "10.0.0.1:29500",
        "WORLD_SIZE": "16",
    }


def test_detects_valid_runpod_cluster_without_guessing_aliases() -> None:
    info = detect_cluster(ClusterConfig(), cluster_env(), verify_interface=False)
    assert info.enabled
    assert info.role == "worker"
    assert info.world_size == 16
    assert info.control_port == 29500
    assert info.rendezvous_port_base == 29501
    assert info.interface == "ens1"


def test_partial_cluster_environment_is_rejected() -> None:
    with pytest.raises(RuntimeValidationError, match="partial RunPod cluster"):
        detect_cluster(
            ClusterConfig(), {"NUM_NODES": "2", "NODE_RANK": "0"}, verify_interface=False
        )


def test_alias_and_world_size_mismatches_are_rejected() -> None:
    bad_alias = cluster_env()
    bad_alias["MASTER_ADDR"] = "10.0.0.9"
    with pytest.raises(RuntimeValidationError, match="disagree"):
        detect_cluster(ClusterConfig(), bad_alias, verify_interface=False)

    bad_world = cluster_env()
    bad_world["WORLD_SIZE"] = "8"
    with pytest.raises(RuntimeValidationError, match="WORLD_SIZE"):
        detect_cluster(ClusterConfig(), bad_world, verify_interface=False)

    bad_primary = cluster_env()
    bad_primary["NODE_RANK"] = "0"
    with pytest.raises(RuntimeValidationError, match="PRIMARY_ADDR must equal NODE_ADDR"):
        detect_cluster(ClusterConfig(), bad_primary, verify_interface=False)


def test_single_node_cluster_contract_is_validated_before_auto_disables_it() -> None:
    env = cluster_env()
    env.update(
        {
            "NODE_RANK": "0",
            "NUM_NODES": "1",
            "NUM_TRAINERS": "8",
            "NODE_ADDR": "10.0.0.1",
            "WORLD_SIZE": "7",
        }
    )
    with pytest.raises(RuntimeValidationError, match="WORLD_SIZE"):
        detect_cluster(ClusterConfig(), env, verify_interface=False)


def test_cluster_ports_may_not_overlap_appliance_listeners() -> None:
    info = detect_cluster(ClusterConfig(), cluster_env(), verify_interface=False)
    with pytest.raises(RuntimeValidationError, match="router and cluster control"):
        validate_cluster_ports(info, 2, reserved_ports={29500: "router"})
    with pytest.raises(RuntimeValidationError, match="engine qwen and cluster rendezvous"):
        validate_cluster_ports(info, 2, reserved_ports={29501: "engine qwen"})


def test_private_coordinator_accepts_only_known_engines() -> None:
    env = cluster_env()
    env["PRIMARY_ADDR"] = env["MASTER_ADDR"] = "127.0.0.1"
    env["HOST_NODE_ADDR"] = "127.0.0.1:29500"
    info = detect_cluster(ClusterConfig(), env, verify_interface=False)
    server = ClusterCoordinator(info, "hash", {"qwen"})
    response = server._process(
        {"op": "poll", "rank": 1, "config_hash": "hash", "engines": {"qwen": "running"}}
    )
    assert response["ok"] is True
    assert server.all_workers_running(["qwen"])
    with pytest.raises(ValueError, match="unknown engine"):
        server._process(
            {
                "op": "poll",
                "rank": 1,
                "config_hash": "hash",
                "engines": {"gemma": "running"},
            }
        )
