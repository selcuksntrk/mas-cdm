"""
Example custom graphs using GraphBuilder.
"""

from backend.app.core.graph.builder import graph_builder


def decision_only_linear():
    nodes = [
        "GetDecision",
        "IdentifyTrigger",
        "AnalyzeRootCause",
        "ScopeDefinition",
        "Drafting",
        "EstablishGoals",
        "IdentifyInformationNeeded",
        "RetrieveInformationNeeded",
        "UpdateDraft",
        "GenerationOfAlternatives",
        "Result",
    ]
    return graph_builder.build(nodes)


if __name__ == "__main__":
    print(decision_only_linear())
