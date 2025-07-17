"""Script that houses common data sampling methods to address class imbalance."""

import abc
import typing as T

import pydantic as pdt
from imblearn import combine, over_sampling, under_sampling


class Sampler(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    pass


class SMOTESampler(Sampler):
    pass


class SMOTETomekSampler(Sampler):
    pass
