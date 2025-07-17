"""Script that houses common data sampling methods to address class imbalance."""

import abc

import pydantic as pdt


class Sampler(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    pass


class SMOTESampler(Sampler):
    pass


class SMOTETomekSampler(Sampler):
    pass
