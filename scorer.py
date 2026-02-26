# scorer.py — Signal scoring and grading

from config import (
    GRADE_A_PLUS_HIGH_SCORE, GRADE_A_PLUS_MEDIUM_SCORE,
    GRADE_MIN_TRADES, GRADE_HIGH_WR, GRADE_MED_WR, GRADE_MIN_EXPECTANCY
)


def compute_score(model_result: dict, backtest_summary: dict) -> dict:
    """
    Compute total score and grade for a signal.
    Returns {"total_score": float, "grade": str, "include": bool, ...}
    """
    model_score      = model_result.get("model_score", 0)
    models_agreeing  = max(model_result.get("models_buy", 0), model_result.get("models_sell", 0))
    agreement_bonus  = (models_agreeing - 3) * 5  # 0, 5, or 10

    wr          = backtest_summary.get("win_rate", 0)
    expectancy  = backtest_summary.get("expectancy", 0)
    total_trades= backtest_summary.get("total_trades", 0)

    if wr >= 65 and expectancy > 1.5:
        backtest_bonus = 10
    elif wr >= 55 and expectancy > 0:
        backtest_bonus = 6
    elif wr >= 45:
        backtest_bonus = 3
    else:
        backtest_bonus = 0

    total_score = min(model_score + agreement_bonus + backtest_bonus, 100)

    # Grade and filter
    include = False
    grade   = "B"

    if (total_score >= GRADE_A_PLUS_HIGH_SCORE and
            total_trades >= GRADE_MIN_TRADES and
            wr >= GRADE_HIGH_WR and
            expectancy > GRADE_MIN_EXPECTANCY):
        grade   = "A+ HIGH"
        include = True
    elif (total_score >= GRADE_A_PLUS_MEDIUM_SCORE and
            total_trades >= GRADE_MIN_TRADES and
            wr >= GRADE_MED_WR and
            expectancy > GRADE_MIN_EXPECTANCY):
        grade   = "A+ MEDIUM"
        include = True

    return {
        "total_score":      round(total_score, 1),
        "model_score":      round(model_score, 1),
        "agreement_bonus":  agreement_bonus,
        "backtest_bonus":   backtest_bonus,
        "grade":            grade,
        "include":          include,
    }
