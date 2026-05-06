"""Behavioral tests for the Routing layer."""

from __future__ import annotations

from pathlib import Path

from tattd_studio.graph.critics import (
    AnatomyCheck,
    PlagiarismCheck,
    QualityScore,
    StyleCoherence,
)
from tattd_studio.graph.routing import (
    CriticVerdicts,
    RoutingThresholds,
    evaluate,
)

THRESHOLDS = RoutingThresholds.from_toml(
    Path(__file__).resolve().parents[2] / "evals" / "calibrated_thresholds.toml"
)


def _verdicts(
    *,
    anatomy_valid: bool = True,
    plagiarism_flagged: bool = False,
    plagiarism_top_sim: float = 0.0,
    style_alignment: float = 0.9,
    quality_aggregate: float = 0.9,
) -> CriticVerdicts:
    return CriticVerdicts(
        anatomy=AnatomyCheck(
            placement_valid=anatomy_valid,
            issues=[] if anatomy_valid else ["bad placement"],
            confidence=0.9,
        ),
        plagiarism=PlagiarismCheck(
            flagged=plagiarism_flagged,
            top_match_artist="Famous Artist" if plagiarism_flagged else "",
            top_match_similarity=plagiarism_top_sim,
            threshold_used=0.85,
            corpus_hit="famous_tattoos_corpus",
        ),
        style=StyleCoherence(
            intent_image_alignment=style_alignment,
            interpretation_notes="ok",
        ),
        quality=QualityScore(
            composition=quality_aggregate,
            linework=quality_aggregate,
            balance=quality_aggregate,
            originality=quality_aggregate,
            notes="ok",
        ),
    )


def test_routing_surfaces_when_all_pass() -> None:
    decision = evaluate(
        _verdicts(),
        thresholds=THRESHOLDS,
        refinement_attempts_so_far=0,
    )
    assert decision.action == "surface"
    assert decision.failed_dimensions == []


def test_routing_refines_on_first_failure() -> None:
    decision = evaluate(
        _verdicts(plagiarism_flagged=True, plagiarism_top_sim=0.92),
        thresholds=THRESHOLDS,
        refinement_attempts_so_far=0,
    )
    assert decision.action == "refine"
    assert "plagiarism" in decision.failed_dimensions
    assert any("diversify" in h for h in decision.refinement_hints)


def test_routing_escalates_on_second_failure() -> None:
    decision = evaluate(
        _verdicts(anatomy_valid=False),
        thresholds=THRESHOLDS,
        refinement_attempts_so_far=1,
    )
    assert decision.action == "escalate"
    assert "anatomy" in decision.failed_dimensions
    assert decision.should_surface is True


def test_routing_flags_each_failing_dimension() -> None:
    decision = evaluate(
        _verdicts(
            anatomy_valid=False,
            style_alignment=0.1,
            quality_aggregate=0.2,
        ),
        thresholds=THRESHOLDS,
        refinement_attempts_so_far=0,
    )
    assert set(decision.failed_dimensions) >= {"anatomy", "style", "quality"}
