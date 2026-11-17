"""Portable, attributed assessments of exact data products."""

from collections.abc import Iterable  # noqa: TC003 -- public signature
from datetime import UTC, datetime
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, ConfigDict, Field, model_validator
from tollan.config.types import PathSegment

from ..core.data_prod import DataProdRef
from ..core.types import FrozenDataModel

__all__ = ["DataProdReview", "current_reviews"]


class DataProdReview(FrozenDataModel):
    """One actor's complete assessment of a product in a QA namespace.

    ``namespace`` identifies the assessment aspect, independently of the product's
    storage namespace. ``role`` records the actor's capacity, such as ``curator``,
    ``diagnostic`` or ``producer``. The actor UUID identifies an existing portable
    Actor; it is not an authentication credential.

    ``flags`` maps local flag names to explicit true/false findings. An empty
    mapping still records a review. A superseding review replaces the complete
    assessment in the same product/actor/role/namespace scope; omitted flags are
    no longer current. Other actors' assessments remain independent. Flag meanings
    and acceptance criteria belong to the namespace's workflow.

    ``previous_uid`` names the immediately preceding review in that same scope,
    or is None for the first review. This model validates fields and rejects a
    self predecessor. DPDB registration checks referenced records and rejects
    inconsistent histories, including conflicting successors. Referenced actors,
    products and predecessors need not accompany the portable record when they
    are already registered. Time and import order do not establish precedence.
    ``run_uid`` optionally identifies the execution that made the assessment.
    ``context`` holds diagnostic evidence, criteria and artifact identifiers used
    in the review. Product artifacts and their locations remain in their index.
    """

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(default_factory=uuid4, strict=False)
    product: DataProdRef
    namespace: PathSegment
    actor_uid: UUID = Field(strict=False)
    role: PathSegment
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC), strict=False
    )
    flags: dict[PathSegment, bool] = Field(default_factory=dict)
    previous_uid: UUID | None = Field(default=None, strict=False)
    note: str = Field(min_length=1)
    run_uid: UUID | None = Field(default=None, strict=False)
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_predecessor(self) -> Self:
        if self.uid == self.previous_uid:
            msg = "Review predecessor must not be itself"
            raise ValueError(msg)
        return self

    @property
    def scope(self) -> tuple[DataProdRef, UUID, str, str]:
        """Product, actor, capacity and assessment namespace."""
        return self.product, self.actor_uid, self.role, self.namespace


def current_reviews(reviews: Iterable[DataProdReview]) -> tuple[DataProdReview, ...]:
    """Select assessments with no successor in the supplied review history.

    Parameters
    ----------
    reviews : iterable of DataProdReview
        Consistent history for the exact products, actors, roles and QA namespaces
        being inspected. This function does not validate history relationships;
        DPDB registration checks them. Include complete history to determine the
        current assessments: an omitted successor cannot be inferred.

    Returns
    -------
    tuple of DataProdReview
        Current assessments, ordered by time and UUID for display. This order
        does not establish precedence. Predecessor links determine which
        assessments remain current, including those with empty flags.
    """
    reviews = tuple(reviews)
    superseded = {review.previous_uid for review in reviews}
    return tuple(
        sorted(
            (review for review in reviews if review.uid not in superseded),
            key=lambda review: (review.created_at, review.uid),
        )
    )
