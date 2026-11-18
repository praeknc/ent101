"""Demand-side data and panel construction (thesis section 4.4).

Second dataset of the thesis: demand for AI models observed at the model,
application and developer/provider level over time, linked to the Part A
price and index panel by model and date.
"""
from . import dictionary, sources  # noqa: F401

__all__ = ["sources", "dictionary", "schema", "crosswalk", "panel", "merge", "coverage", "docs"]
