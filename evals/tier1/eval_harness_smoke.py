"""Tier 1 Eval Harness smoke test.

Tracer bullet for the Eval Harness wiring (CONTEXT.md → "Eval Harness").
Uses a deterministic, LLM-free DeepEval metric so CI can run this slice
without provider credentials.
"""

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class _KeywordPresenceMetric(BaseMetric):
    """Deterministic metric: scores 1.0 iff the keyword appears in the output."""

    def __init__(self, keyword: str, threshold: float = 0.5) -> None:
        self.keyword = keyword
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output or ""
        self.score = 1.0 if self.keyword.lower() in output.lower() else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"keyword '{self.keyword}' "
            f"{'found' if self.success else 'missing'} in actual_output"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "KeywordPresenceMetric"


def test_eval_harness_smoke() -> None:
    case = LLMTestCase(
        input="Describe the artifact this repo ships.",
        actual_output="Tattd Studio is a multi-turn tattoo design agent.",
    )
    assert_test(case, [_KeywordPresenceMetric(keyword="tattoo")])
