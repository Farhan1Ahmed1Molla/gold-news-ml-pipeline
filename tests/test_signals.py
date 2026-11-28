"""
tests/test_signals.py — Automated Unit Tests for signals.py.

Run with:
    uv run pytest -v
"""

import math

import numpy as np
import pytest
from signals import (
    PRICE_THRESHOLD_PCT,
    SENTIMENT_THRESHOLD,
    VALID_SIGNALS,
    generate_signal,
)


# ==============================================================================
# 1. ACCUMULATE SIGNAL TESTS (Exact Boundary & Above)
# ==============================================================================
def test_accumulate_at_exact_boundary():
    """Meeting both thresholds at the exact boundary (>= 0.5 and >= 0.15) returns ACCUMULATE."""
    signal = generate_signal(
        predicted_change_pct=PRICE_THRESHOLD_PCT,  # 0.5
        sentiment_score=SENTIMENT_THRESHOLD,  # 0.15
        article_count=1,
    )
    assert signal == "ACCUMULATE"
    assert signal in VALID_SIGNALS


def test_accumulate_strong_bullish():
    """Values well above thresholds with multiple articles return ACCUMULATE."""
    signal = generate_signal(
        predicted_change_pct=1.85, sentiment_score=0.62, article_count=12
    )
    assert signal == "ACCUMULATE"


# ==============================================================================
# 2. PAUSE SIGNAL TESTS (Exact Boundary & Below)
# ==============================================================================
def test_pause_at_exact_boundary():
    """Meeting both negative thresholds at the exact boundary (<= -0.5 and <= -0.15) returns PAUSE."""
    signal = generate_signal(
        predicted_change_pct=-PRICE_THRESHOLD_PCT,  # -0.5
        sentiment_score=-SENTIMENT_THRESHOLD,  # -0.15
        article_count=1,
    )
    assert signal == "PAUSE"


def test_pause_strong_bearish():
    """Strong negative price prediction paired with bearish news returns PAUSE."""
    signal = generate_signal(
        predicted_change_pct=-2.10, sentiment_score=-0.55, article_count=8
    )
    assert signal == "PAUSE"


# ==============================================================================
# 3. HOLD SIGNAL TESTS (Missed Thresholds & Conflicting Signals)
# ==============================================================================
@pytest.mark.parametrize(
    "pct, sentiment, count",
    [
        (0.49, 0.15, 5),  # Price change just below +0.5% threshold
        (0.50, 0.14, 5),  # Sentiment just below +0.15 threshold
        (-0.49, -0.15, 5),  # Price drop just above -0.5% threshold
        (-0.50, -0.14, 5),  # Sentiment just above -0.15 threshold
        (1.20, -0.40, 5),  # Conflicting: strong price up, but bearish news
        (-1.20, 0.40, 5),  # Conflicting: strong price down, but bullish news
        (0.0, 0.0, 5),  # Completely flat market and neutral news
    ],
)
def test_hold_on_missed_or_conflicting_thresholds(
    pct: float, sentiment: float, count: int
):
    """Any case where either threshold is missed or signals conflict must return HOLD."""
    assert generate_signal(pct, sentiment, count) == "HOLD"


# ==============================================================================
# 4. ZERO-ARTICLE SAFETY GUARDRAIL
# ==============================================================================
@pytest.mark.parametrize(
    "pct, sentiment",
    [
        (2.50, 0.80),  # Would otherwise be ACCUMULATE
        (-2.50, -0.80),  # Would otherwise be PAUSE
        (0.0, 0.0),  # Neutral
    ],
)
def test_zero_articles_always_forces_hold(pct: float, sentiment: float):
    """When article_count == 0, generate_signal MUST return HOLD regardless of inputs."""
    assert generate_signal(pct, sentiment, article_count=0) == "HOLD"


# ==============================================================================
# 5. DEFENSIVE INPUT VALIDATION TESTS
# ==============================================================================
@pytest.mark.parametrize("bad_float", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_floats(bad_float: float):
    """NaN and Infinity must raise ValueError instead of silently producing HOLD."""
    with pytest.raises(ValueError):
        generate_signal(bad_float, 0.2, 5)
    with pytest.raises(ValueError):
        generate_signal(0.6, bad_float, 5)


@pytest.mark.parametrize("bad_bool", [True, False])
def test_rejects_booleans(bad_bool: bool):
    """Booleans must raise TypeError even though bool is a subclass of int in Python."""
    with pytest.raises(TypeError):
        generate_signal(bad_bool, 0.2, 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        generate_signal(0.6, bad_bool, 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        generate_signal(0.6, 0.2, bad_bool)  # type: ignore[arg-type]


def test_rejects_negative_or_float_article_count():
    """article_count must be a non-negative integer."""
    with pytest.raises(ValueError):
        generate_signal(0.6, 0.2, -1)
    with pytest.raises(TypeError):
        generate_signal(0.6, 0.2, 3.5)  # type: ignore[arg-type]


def test_accepts_numpy_numeric_scalars_and_rejects_numpy_bool():
    """NumPy numeric scalars from pandas/sklearn work while np.bool_ and NaN fail."""
    assert generate_signal(np.float64(0.6), np.float32(0.2), np.int64(4)) == "ACCUMULATE"
    assert generate_signal(np.float32(-0.6), np.float64(-0.2), np.int32(3)) == "PAUSE"

    with pytest.raises(TypeError):
        generate_signal(0.6, 0.2, np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_signal(np.float64("nan"), 0.2, np.int64(2))
    with pytest.raises(ValueError):
        generate_signal(0.6, 0.2, np.int64(-1))
