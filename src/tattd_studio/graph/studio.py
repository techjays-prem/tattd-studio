"""Studio graph — full four-Critic pipeline with Routing and Refinement.

Slice #7 extends the slice-#6 graph with three more Critics running in
parallel branches plus a Refinement loop that re-runs Generation once
when any Critic flags. A second flag escalates: surface to the human
client with verdicts attached.

Topology:

    START
      → trace_start
      → consultation
      → generation
      → fanout (paragraph node — runs all four Critics in parallel by
        adding parallel edges that converge into routing)
      → routing
      → if "refine" → consultation (loop back) ; else → trace_end → END

LangGraph 1.0.x runs independent branches in parallel when multiple
nodes share a common predecessor; we use that shape directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from tattd_studio.generation import GenerationClient
from tattd_studio.graph.critics import (
    AnatomyCritic,
    PlagiarismCritic,
    QualityCritic,
    StyleCritic,
)
from tattd_studio.graph.routing import (
    CriticVerdicts,
    RoutingThresholds,
    evaluate,
)
from tattd_studio.graph.state import StudioState
from tattd_studio.knowledge import KnowledgeRetriever
from tattd_studio.models import Intent


def build_studio_graph(
    *,
    retriever: KnowledgeRetriever,
    generation_client: GenerationClient,
    anatomy_critic: AnatomyCritic,
    plagiarism_critic: PlagiarismCritic | None = None,
    style_critic: StyleCritic | None = None,
    quality_critic: QualityCritic | None = None,
    thresholds: RoutingThresholds | None = None,
    knowledge_top_k: int = 5,
    refine_intent: Callable[[Intent, list[str]], Intent] | None = None,
) -> Any:
    """Compile and return the Studio graph.

    All four Critics are optional so slice-#6 callers (Anatomy only) keep
    working. When any Critic is omitted, its verdict slot stays empty
    and Routing ignores that dimension.
    """

    refine_intent = refine_intent or _default_refine_intent

    def trace_start(state: StudioState) -> dict[str, Any]:
        meta = dict(state.get("metadata") or {})
        meta["_started_at"] = time.monotonic()
        return {"metadata": meta}

    def consultation_node(state: StudioState) -> dict[str, Any]:
        chunks = retriever.retrieve(state["intent"].refined_description, k=knowledge_top_k)
        return {"knowledge_chunks": chunks}

    def generation_node(state: StudioState) -> dict[str, Any]:
        candidates = generation_client.generate(state["intent"], n=state["n_candidates"])
        return {"candidate_designs": candidates}

    def anatomy_node(state: StudioState) -> dict[str, Any]:
        verdicts = [
            anatomy_critic.check(cd, state["placement_context"])
            for cd in state["candidate_designs"]
        ]
        return {"anatomy_checks": verdicts}

    def plagiarism_node(state: StudioState) -> dict[str, Any]:
        if plagiarism_critic is None:
            return {"plagiarism_checks": []}
        verdicts = [plagiarism_critic.check(cd) for cd in state["candidate_designs"]]
        return {"plagiarism_checks": verdicts}

    def style_node(state: StudioState) -> dict[str, Any]:
        if style_critic is None:
            return {"style_checks": []}
        verdicts = [
            style_critic.check(state["intent"], cd) for cd in state["candidate_designs"]
        ]
        return {"style_checks": verdicts}

    def quality_node(state: StudioState) -> dict[str, Any]:
        if quality_critic is None:
            return {"quality_checks": []}
        verdicts = [quality_critic.check(cd) for cd in state["candidate_designs"]]
        return {"quality_checks": verdicts}

    def routing_node(state: StudioState) -> dict[str, Any]:
        if thresholds is None:
            return {"routing_decisions": []}
        n = len(state["candidate_designs"])
        plagiarism = state.get("plagiarism_checks") or [None] * n
        style = state.get("style_checks") or [None] * n
        quality = state.get("quality_checks") or [None] * n
        decisions = []
        for i in range(n):
            if (
                plagiarism[i] is None or style[i] is None or quality[i] is None
            ):
                # Skip routing for this candidate — Critic missing.
                continue
            verdicts = CriticVerdicts(
                anatomy=state["anatomy_checks"][i],
                plagiarism=plagiarism[i],
                style=style[i],
                quality=quality[i],
            )
            decisions.append(
                evaluate(
                    verdicts,
                    thresholds=thresholds,
                    refinement_attempts_so_far=state.get("refinement_attempts", 0),
                )
            )
        return {"routing_decisions": decisions}

    def refinement_branch(state: StudioState) -> str:
        decisions = state.get("routing_decisions") or []
        if not decisions:
            return "trace_end"
        if state.get("refinement_attempts", 0) >= 1:
            return "trace_end"
        if any(d.action == "refine" for d in decisions):
            return "refine_intent"
        return "trace_end"

    def refine_intent_node(state: StudioState) -> dict[str, Any]:
        decisions = state.get("routing_decisions") or []
        hints = [h for d in decisions for h in d.refinement_hints]
        new_intent = refine_intent(state["intent"], hints)
        return {
            "intent": new_intent,
            "refinement_attempts": state.get("refinement_attempts", 0) + 1,
        }

    def trace_end(state: StudioState) -> dict[str, Any]:
        meta = dict(state.get("metadata") or {})
        started = meta.pop("_started_at", time.monotonic())
        meta["latency_seconds"] = max(0.0, time.monotonic() - started)
        meta["candidate_count"] = len(state["candidate_designs"])
        meta["chunk_count"] = len(state["knowledge_chunks"])
        meta["refinement_attempts"] = state.get("refinement_attempts", 0)
        return {"metadata": meta}

    builder: StateGraph = StateGraph(StudioState)
    builder.add_node("trace_start", trace_start)
    builder.add_node("consultation", consultation_node)
    builder.add_node("generation", generation_node)
    builder.add_node("anatomy_critic", anatomy_node)
    builder.add_node("plagiarism_critic", plagiarism_node)
    builder.add_node("style_critic", style_node)
    builder.add_node("quality_critic", quality_node)
    builder.add_node("routing", routing_node)
    builder.add_node("refine_intent", refine_intent_node)
    builder.add_node("trace_end", trace_end)

    builder.add_edge(START, "trace_start")
    builder.add_edge("trace_start", "consultation")
    builder.add_edge("consultation", "generation")
    # All four Critics run in parallel from `generation`.
    builder.add_edge("generation", "anatomy_critic")
    builder.add_edge("generation", "plagiarism_critic")
    builder.add_edge("generation", "style_critic")
    builder.add_edge("generation", "quality_critic")
    # All four converge into routing.
    builder.add_edge("anatomy_critic", "routing")
    builder.add_edge("plagiarism_critic", "routing")
    builder.add_edge("style_critic", "routing")
    builder.add_edge("quality_critic", "routing")
    # Routing decides Refinement vs surface.
    builder.add_conditional_edges(
        "routing", refinement_branch, {
            "refine_intent": "refine_intent",
            "trace_end": "trace_end",
        }
    )
    # Refinement loops back into Generation directly (Consultation context
    # carries; we only re-generate against the refined Intent).
    builder.add_edge("refine_intent", "generation")
    builder.add_edge("trace_end", END)

    return builder.compile()


def _default_refine_intent(intent: Intent, hints: list[str]) -> Intent:
    """Append refinement hints into the refined_description.

    Slice #7 keeps this trivial; Consultation in slice #6 handled the
    structured-Intent shape, and a richer refinement (style retag,
    placement adjust, motif diversify) lands when the Studio's product
    surface matures.
    """
    suffix = ""
    if hints:
        suffix = " " + "; ".join(f"[{h}]" for h in hints)
    return Intent(refined_description=intent.refined_description + suffix)
