"""Portable assessments validate local facts and select linked current records."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from tolteca_datamodels.data_prod import (
    DataProd,
    DataProdIndex,
    DataProdRef,
    DataProdReview,
    current_reviews,
)
from tolteca_datamodels.data_prod.metadata import ScienceMeta


@pytest.fixture
def review_fields():
    return {
        "product": DataProdRef(uid="analysis", namespace_uid=uuid4()),
        "actor_uid": uuid4(),
        "namespace": "quality",
        "role": "curator",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "note": "Checked the recorded result.",
    }


def test_review_rejects_self_predecessor(review_fields):
    uid = uuid4()
    with pytest.raises(ValidationError, match="must not be itself"):
        DataProdReview.model_validate(
            {**review_fields, "uid": uid, "previous_uid": uid}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flags", {"needs_followup": "true"}),
        ("note", ""),
        ("created_at", datetime(2026, 1, 1)),  # noqa: DTZ001 -- reject a naive timestamp
    ],
)
def test_review_rejects_invalid_field(review_fields, field, value):
    with pytest.raises(ValidationError) as exc:
        DataProdReview.model_validate({**review_fields, field: value})
    assert any(error["loc"][0] == field for error in exc.value.errors())


@pytest.mark.parametrize("changed_field", ["uid", "namespace_uid"])
def test_index_reviews_require_its_exact_product(review_fields, changed_field):
    review = DataProdReview.model_validate(review_fields)
    product = DataProd(
        uid=review.product.uid,
        namespace_uid=review.product.namespace_uid,
        meta=ScienceMeta(name="Analysis", data_prod_type="dp_science"),
    )
    assert DataProdIndex(product=product, reviews=[review]).reviews == [review]
    wrong_reference = DataProdRef.model_validate(
        {
            **review.product.model_dump(),
            changed_field: "another-analysis" if changed_field == "uid" else uuid4(),
        }
    )
    wrong_review = DataProdReview.model_validate(
        {**review_fields, "product": wrong_reference}
    )
    with pytest.raises(ValidationError, match="exact product"):
        DataProdIndex(product=product, reviews=[wrong_review])


def test_current_reviews_follow_links_and_sort_only_for_display(review_fields):
    first = DataProdReview.model_validate(
        {**review_fields, "flags": {"needs_followup": True, "missing_artifacts": False}}
    )
    earlier = first.created_at - timedelta(hours=1)
    second = DataProdReview.model_validate(
        {
            **review_fields,
            "uid": UUID(int=1),
            "previous_uid": first.uid,
            "created_at": earlier,
            "flags": {"needs_followup": False},
        }
    )
    independent = DataProdReview.model_validate(
        {
            **review_fields,
            "uid": UUID(int=2),
            "actor_uid": uuid4(),
            "created_at": earlier,
            "flags": {},
        }
    )

    result = current_reviews(iter([independent, second, first]))

    assert result == (second, independent)
    assert result[0].flags == {"needs_followup": False}
    assert result[1].flags == {}


@pytest.mark.parametrize("field", ["product", "actor_uid", "role", "namespace"])
def test_current_reviews_keep_independent_scopes(review_fields, field):
    first = DataProdReview.model_validate(review_fields)
    changes = {
        "product": DataProdRef(uid=first.product.uid, namespace_uid=uuid4()),
        "actor_uid": uuid4(),
        "role": "diagnostic",
        "namespace": "calibration",
    }
    independent = DataProdReview.model_validate(
        {
            **review_fields,
            field: changes[field],
            "created_at": first.created_at + timedelta(seconds=1),
        }
    )

    assert current_reviews([independent, first]) == (first, independent)


def test_current_reviews_only_consider_supplied_history(review_fields):
    first = DataProdReview.model_validate(review_fields)
    second = DataProdReview.model_validate({**review_fields, "previous_uid": first.uid})

    assert current_reviews([]) == ()
    assert current_reviews([first]) == (first,)
    assert current_reviews([second]) == (second,)
