"""Grounded citations.

The model may attach citations to an answer, but a citation's source_id MUST be one the server
offered this turn — the model may not invent or alter a source (AD-7: answers stay grounded in
sources we actually gave it). Like tool calls, citations cross an untrusted boundary: the offered
id set is server-owned, the citations are the model's proposal.
"""

from collections.abc import Iterable
from typing import Annotated

from pydantic import Field

from stanok_core.tools import ContractModel

# A citation id is opaque and server-assigned; constrain it so empty/garbage fails at parse time.
CitationId = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")]
Quote = Annotated[str, Field(min_length=1, max_length=2000)]


class Citation(ContractModel):
    """A citation the model attached to an answer (untrusted until source_id is checked)."""

    source_id: CitationId
    quote: Quote


def ungrounded_citations(citations: Iterable[Citation], offered_ids: frozenset[str]) -> list[str]:
    """Return the source_ids that were not offered (empty list = every citation grounded).

    offered_ids is the server-owned set of sources shown to the model this turn. A non-empty result
    means the model cited something it was never given; the answer must not go out as grounded.
    """
    return [c.source_id for c in citations if c.source_id not in offered_ids]
