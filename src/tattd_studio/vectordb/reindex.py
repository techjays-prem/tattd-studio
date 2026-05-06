"""Alias-swap rebuild script.

Rebuilds a Vector Store collection behind a stable alias. Callers supply a
``populate`` function that fills a freshly-created collection with points;
this script wires it through ``VectorStore.alias_swap_rebuild`` so readers
querying the alias never see an empty index during the swap.

Used as a module API by the slices that own each Vector Store collection:

- Knowledge Corpus rebuilds (slice #3)
- Artist Portfolio Index rebuilds (slice #8)
- Famous Tattoos Corpus rebuilds (slice #7)

Each owner provides its own ``populate`` function and calls ``rebuild``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from tattd_studio.vectordb.qdrant_client import VectorStore


def rebuild(
    store: VectorStore,
    alias: str,
    populate: Callable[[VectorStore, str], None],
    timestamp: dt.datetime | None = None,
) -> str:
    """Run a full alias-swap rebuild and return the new collection name.

    The new collection name is derived from the alias plus a UTC timestamp
    so successive rebuilds produce monotonically increasing names that
    are auditable in Qdrant logs.
    """
    ts = (timestamp or dt.datetime.now(dt.UTC)).strftime("%Y%m%dT%H%M%S")
    new_collection = f"{alias}_{ts}"

    def _build(name: str) -> None:
        store.create_collection(name)
        populate(store, name)

    store.alias_swap_rebuild(
        alias=alias, new_collection=new_collection, build_fn=_build
    )
    return new_collection
