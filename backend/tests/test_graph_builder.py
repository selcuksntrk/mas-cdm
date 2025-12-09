import pytest

from backend.app.core.graph.builder import GraphBuilder, GraphValidationError


def test_valid_linear_graph():
    builder = GraphBuilder()
    nodes = ["GetDecision", "IdentifyTrigger", "AnalyzeRootCause"]
    built = builder.build(nodes)
    assert built["nodes"] == nodes
    assert built["edges"][0] == ("GetDecision", "IdentifyTrigger")


def test_unknown_node_fails():
    builder = GraphBuilder()
    with pytest.raises(GraphValidationError):
        builder.build(["GetDecision", "BogusNode"])


def test_cycle_detection():
    builder = GraphBuilder()
    nodes = ["GetDecision", "IdentifyTrigger"]
    edges = [("GetDecision", "IdentifyTrigger"), ("IdentifyTrigger", "GetDecision")]
    with pytest.raises(GraphValidationError):
        builder.build(nodes, edges)


def test_unreachable_detection():
    builder = GraphBuilder()
    nodes = ["GetDecision", "IdentifyTrigger", "AnalyzeRootCause"]
    edges = [("GetDecision", "IdentifyTrigger")]
    with pytest.raises(GraphValidationError):
        builder.build(nodes, edges)
