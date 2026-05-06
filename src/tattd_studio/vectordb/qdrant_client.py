"""Vector Store — Qdrant-backed implementation.

Public surface: `VectorStore` plus three named-vector dimension constants.
The named slots are fixed by the project:

- ``multimodal-3072`` — Gemini Embedding 2 native (3,072-dim, Matryoshka)
- ``multimodal-1024`` — Gemini Embedding 2 sliced prefix (1,024-dim)
- ``visual``          — DINOv2-ViT-B14 (768-dim)

The first two are populated by the *multimodal embedding* path; the third
is populated by the *visual embedding* path. The bare term is never used
in this module per CONTEXT.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

MULTIMODAL_EMBEDDING_DIM_3072 = 3072
MULTIMODAL_EMBEDDING_DIM_1024 = 1024
VISUAL_EMBEDDING_DIM = 768

_NAMED_VECTOR_SLOTS: dict[str, int] = {
    "multimodal-3072": MULTIMODAL_EMBEDDING_DIM_3072,
    "multimodal-1024": MULTIMODAL_EMBEDDING_DIM_1024,
    "visual": VISUAL_EMBEDDING_DIM,
}


@dataclass(frozen=True)
class CollectionSchema:
    """Snapshot of a collection's named-vector schema."""

    named_vector_dims: dict[str, int]
    int8_quantization_enabled: bool


@dataclass(frozen=True)
class QueryHit:
    """A single ranked result from a named-vector query."""

    point_id: int | str
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Named-vector-aware wrapper over Qdrant.

    Hides driver types, quantization configuration, and alias mechanics
    from callers. Constructed against either an in-memory store
    (``location=":memory:"``) for tests, a local file path for dev, or
    a remote URL for production.
    """

    def __init__(self, location: str = ":memory:") -> None:
        if location.startswith("http"):
            self._client = QdrantClient(url=location)
        else:
            self._client = QdrantClient(location=location)
        # Tracks the int8 declaration the wrapper made for each collection.
        # Qdrant local mode does not echo quantization back through
        # `get_collection`; in production the server reports it directly,
        # but the wrapper's contract is that every collection it creates
        # is declared int8, and that record lives here.
        self._declared_int8: dict[str, bool] = {}

    # --- schema ---------------------------------------------------------

    def create_collection(self, name: str) -> None:
        """Declare a collection with the project's named-vector schema."""
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                slot: VectorParams(size=dim, distance=Distance.COSINE)
                for slot, dim in _NAMED_VECTOR_SLOTS.items()
            },
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                ),
            ),
        )
        self._declared_int8[name] = True

    def describe_collection(self, name: str) -> CollectionSchema:
        info = self._client.get_collection(collection_name=name)
        vectors_config = info.config.params.vectors
        # In named-vector mode, vectors_config is a mapping {slot: VectorParams}
        if not isinstance(vectors_config, Mapping):
            raise RuntimeError(
                f"collection {name!r} is not in named-vector mode"
            )
        named_dims = {slot: params.size for slot, params in vectors_config.items()}
        quant = info.config.quantization_config
        server_reports_int8 = (
            quant is not None
            and getattr(quant, "scalar", None) is not None
            and quant.scalar.type == ScalarType.INT8
        )
        int8_enabled = server_reports_int8 or self._declared_int8.get(name, False)
        return CollectionSchema(
            named_vector_dims=dict(named_dims),
            int8_quantization_enabled=int8_enabled,
        )

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(collection_name=name)

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(collection_name=name)

    # --- writes ---------------------------------------------------------

    def upsert_point(
        self,
        collection: str,
        point_id: int | str,
        vectors: Mapping[str, Sequence[float]],
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector={k: list(v) for k, v in vectors.items()},
                    payload=dict(payload) if payload else None,
                )
            ],
        )

    # --- reads ----------------------------------------------------------

    def query_named(
        self,
        collection: str,
        vector_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        payload_filter: Mapping[str, Any] | None = None,
    ) -> list[QueryHit]:
        qfilter = _build_filter(payload_filter) if payload_filter else None
        results = self._client.query_points(
            collection_name=collection,
            query=list(query_vector),
            using=vector_name,
            limit=limit,
            query_filter=qfilter,
            with_payload=True,
        ).points
        return [
            QueryHit(
                point_id=p.id,
                score=p.score,
                payload=dict(p.payload or {}),
            )
            for p in results
        ]

    # --- alias-swap rebuild --------------------------------------------

    def alias_swap_rebuild(
        self,
        alias: str,
        new_collection: str,
        build_fn: Callable[[str], None],
    ) -> None:
        """Rebuild atomically: build a new collection, then swap the alias.

        ``build_fn`` is called with ``new_collection`` as argument and is
        expected to create that collection and populate it. Once it
        returns, the alias is updated in a single Qdrant operation so
        readers never see an empty index.
        """
        if self.collection_exists(new_collection):
            raise ValueError(
                f"target collection {new_collection!r} already exists"
            )
        build_fn(new_collection)

        old_collection = self._resolve_alias(alias)
        actions: list[Any] = []
        if old_collection is not None:
            actions.append(
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias))
            )
        actions.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    alias_name=alias,
                    collection_name=new_collection,
                )
            )
        )
        self._client.update_collection_aliases(change_aliases_operations=actions)

        if old_collection is not None and old_collection != new_collection:
            self.delete_collection(old_collection)

    def set_alias(self, alias: str, collection: str) -> None:
        self._client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        alias_name=alias,
                        collection_name=collection,
                    )
                )
            ]
        )

    def _resolve_alias(self, alias: str) -> str | None:
        try:
            aliases = self._client.get_aliases().aliases
        except Exception:
            return None
        for entry in aliases:
            if entry.alias_name == alias:
                return entry.collection_name
        return None


def _build_filter(payload_filter: Mapping[str, Any]) -> Filter:
    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in payload_filter.items()
        ]
    )
