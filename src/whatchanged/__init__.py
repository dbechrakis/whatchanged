"""Which segment explains a change in a metric."""

from whatchanged._version import __version__
from whatchanged.compare import Comparison, compare
from whatchanged.errors import WhatChangedError
from whatchanged.metrics import Mean, Ratio, Sum, parse_metric
from whatchanged.result import (
    CandidateSet,
    Change,
    Decomposition,
    Explanation,
    NoiseCheck,
    Segment,
)

__all__ = [
    "CandidateSet",
    "Change",
    "Comparison",
    "Decomposition",
    "Explanation",
    "Mean",
    "NoiseCheck",
    "Ratio",
    "Segment",
    "Sum",
    "WhatChangedError",
    "__version__",
    "compare",
    "parse_metric",
]
