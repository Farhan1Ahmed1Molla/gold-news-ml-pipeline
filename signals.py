"""Pure long-term gold accumulation signal rules (no network or database I/O)."""

import math
import numbers

# Price move must reach +/-0.5% to trigger an actionable signal.
PRICE_THRESHOLD_PCT: float = 0.5

# News sentiment must reach +/-0.15 to confirm the predicted price move.
SENTIMENT_THRESHOLD: float = 0.15

# Allowed output signal values.
VALID_SIGNALS: frozenset[str] = frozenset({"ACCUMULATE", "HOLD", "PAUSE"})


def validate_finite_float(value: object, name: str) -> float:
    """Validate that value is a finite real number (excluding bool) and return float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a finite int or float, got {type(value).__name__}"
        )
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return numeric_value


def validate_non_negative_int(value: object, name: str) -> int:
    """Validate that value is a non-negative integer (excluding bool) and return int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f"{name} must be >= 0, got {int_value}")
    return int_value


def generate_signal(
    predicted_change_pct: float, sentiment_score: float, article_count: int
) -> str:
    """Classify a daily gold forecast into ACCUMULATE, HOLD, or PAUSE."""
    clean_pct = validate_finite_float(predicted_change_pct, "predicted_change_pct")
    clean_sentiment = validate_finite_float(sentiment_score, "sentiment_score")
    clean_count = validate_non_negative_int(article_count, "article_count")

    if clean_count == 0:
        return "HOLD"

    if clean_pct >= PRICE_THRESHOLD_PCT and clean_sentiment >= SENTIMENT_THRESHOLD:
        return "ACCUMULATE"

    if clean_pct <= -PRICE_THRESHOLD_PCT and clean_sentiment <= -SENTIMENT_THRESHOLD:
        return "PAUSE"

    return "HOLD"
