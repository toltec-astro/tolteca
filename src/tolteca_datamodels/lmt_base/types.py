"""Common field value types for LMT data."""

from pydantic import AwareDatetime, NonNegativeInt

__all__ = ["DatetimeT", "LabelT", "SeqNumT"]

type LabelT = str
type SeqNumT = NonNegativeInt
type DatetimeT = AwareDatetime
