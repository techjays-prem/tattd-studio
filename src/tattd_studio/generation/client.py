"""Generation Client implementation.

Wraps Gemini Nano Banana 2 / Pro at runtime. Hides retry, in-memory cache,
and prompt versioning. The actual provider call is injected via
``generate_fn`` so tests are deterministic and CI never makes live calls;
the default `generate_fn` builds a real Gemini client at first use.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from typing import Protocol

from tattd_studio.models import Intent
from tattd_studio.models.candidate_design import CandidateDesign

PROMPT_TEMPLATE_VERSION = "v1.0"


class GenerationError(Exception):
    """Non-transient failure from the generation provider."""


class TransientGenerationError(GenerationError):
    """Transient failure that the client may retry."""


class _GenerateFn(Protocol):
    def __call__(self, prompt: str, n: int) -> list[str]: ...


def build_prompt(intent: Intent) -> str:
    """Render an Intent into a provider-facing prompt string.

    Uses ``PROMPT_TEMPLATE_VERSION`` as the implicit contract: any change
    to the template body MUST bump the version so generations are traceable
    to the exact template that produced them.
    """
    return f"[tattd-studio/{PROMPT_TEMPLATE_VERSION}] {intent.refined_description}"


class GenerationClient:
    """Returns N Candidate Designs for a refined Intent.

    Args:
        source_model_id: Identifier carried into every CandidateDesign so
            downstream Critics can attribute output to the runtime
            generation provider that produced it.
        generate_fn: The actual call into the provider. Injectable for
            tests; in production a closure over `google.genai.Client` lives
            here.
        max_retries: Bounded retry attempts for transient failures.
        sleep: Sleep function for retry back-off (injectable for tests).
        clock: Returns the current time. Injectable for tests.
    """

    def __init__(
        self,
        *,
        source_model_id: str,
        generate_fn: _GenerateFn,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self._source_model_id = source_model_id
        self._generate_fn = generate_fn
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._cache: dict[tuple[str, str], list[CandidateDesign]] = {}

    def generate(self, intent: Intent, n: int) -> list[CandidateDesign]:
        if n <= 0:
            raise ValueError("n must be positive")
        prompt = build_prompt(intent)
        cache_key = (prompt, self._source_model_id)
        cached = self._cache.get(cache_key)
        if cached is not None and len(cached) >= n:
            return list(cached[:n])

        image_uris = self._call_with_retry(prompt, n)
        now = self._clock()
        candidates = [
            CandidateDesign(
                image_uri=uri,
                prompt=prompt,
                source_model_id=self._source_model_id,
                metadata={"prompt_template_version": PROMPT_TEMPLATE_VERSION},
                created_at=now,
            )
            for uri in image_uris
        ]
        self._cache[cache_key] = candidates
        return candidates

    def _call_with_retry(self, prompt: str, n: int) -> list[str]:
        attempts = 0
        backoff = 0.05
        while True:
            try:
                return self._generate_fn(prompt, n)
            except TransientGenerationError:
                attempts += 1
                if attempts >= self._max_retries:
                    raise
                self._sleep(backoff * (2 ** (attempts - 1)))
