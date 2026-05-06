"""Tier 3 stretch — synthetic test-set generator.

Per the implementation plan's Eval Surface table, the Tier 3 stretch is
golden-set expansion via DeepEval's Synthesizer. This module ships:

- ``expand_anatomy_cases(seed, n)`` — deterministic permutation
  augmenter (CI baseline). Generates variants by swapping body parts,
  scaling sizes, and toggling modifiers; produces ``n`` augmented cases
  from the seed Golden Set.
- ``synthesize_with_judge(seed, n)`` — live wrapper around DeepEval's
  ``Synthesizer.generate_goldens_from_contexts(...)`` that produces
  semantically novel cases. Gated by ``RUN_LIVE_SYNTHESIZER=1``.

Both produce records in the same shape as the original seed so the
expanded set can drop into any of the Tier 1 evals' Golden Set paths.
"""

from tattd_studio.eval_synth.augment import (
    expand_anatomy_cases,
    synthesize_with_judge,
)

__all__ = [
    "expand_anatomy_cases",
    "synthesize_with_judge",
]
