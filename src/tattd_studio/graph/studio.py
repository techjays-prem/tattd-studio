"""Minimal Studio graph: Consultation → Generation → Anatomy Critic → END.

This slice (#6) wires the three building blocks shipped in #3, #4, #5
into a runnable LangGraph state machine. No Refinement loop yet —
failure verdicts surface alongside Candidate Designs.

Per IMPLEMENTATION_PLAN.md → Risks & Gotchas, LangGraph 1.0.x is pinned
tight in pyproject.toml; the API surface is held constant by exercising
only `StateGraph`, `START`, `END`, and `add_node`/`add_edge`.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from tattd_studio.generation import GenerationClient
from tattd_studio.graph.critics import AnatomyCritic
from tattd_studio.graph.state import StudioState
from tattd_studio.knowledge import KnowledgeRetriever


def build_studio_graph(
    *,
    retriever: KnowledgeRetriever,
    generation_client: GenerationClient,
    anatomy_critic: AnatomyCritic,
    knowledge_top_k: int = 5,
) -> Any:
    """Compile and return the Studio graph.

    Returns a CompiledStateGraph; callers invoke ``graph.invoke(state)``.
    """

    def consultation_node(state: StudioState) -> StudioState:
        chunks = retriever.retrieve(state["intent"].refined_description, k=knowledge_top_k)
        return {**state, "knowledge_chunks": chunks}

    def generation_node(state: StudioState) -> StudioState:
        candidates = generation_client.generate(state["intent"], n=state["n_candidates"])
        return {**state, "candidate_designs": candidates}

    def anatomy_node(state: StudioState) -> StudioState:
        verdicts = [
            anatomy_critic.check(cd, state["placement_context"])
            for cd in state["candidate_designs"]
        ]
        return {**state, "anatomy_checks": verdicts}

    def trace_start(state: StudioState) -> StudioState:
        meta = dict(state.get("metadata") or {})
        meta["_started_at"] = time.monotonic()
        return {**state, "metadata": meta}

    def trace_end(state: StudioState) -> StudioState:
        meta = dict(state.get("metadata") or {})
        started = meta.pop("_started_at", time.monotonic())
        meta["latency_seconds"] = max(0.0, time.monotonic() - started)
        meta["candidate_count"] = len(state["candidate_designs"])
        meta["chunk_count"] = len(state["knowledge_chunks"])
        return {**state, "metadata": meta}

    builder: StateGraph = StateGraph(StudioState)
    builder.add_node("trace_start", trace_start)
    builder.add_node("consultation", consultation_node)
    builder.add_node("generation", generation_node)
    builder.add_node("anatomy_critic", anatomy_node)
    builder.add_node("trace_end", trace_end)

    builder.add_edge(START, "trace_start")
    builder.add_edge("trace_start", "consultation")
    builder.add_edge("consultation", "generation")
    builder.add_edge("generation", "anatomy_critic")
    builder.add_edge("anatomy_critic", "trace_end")
    builder.add_edge("trace_end", END)

    return builder.compile()
