# performance metrics for a per-bar pnl series.

import math

import numpy as np


def bars_per_year(index):
    """Bar density per year inferred from a datetime index."""

    years = (index.max() - index.min()).total_seconds() / (365.25 * 86400)
    if years <= 0:
        return np.nan
    return len(index) / years


def sharpe(pnl):
    """Annualized Sharpe ratio of a per-bar pnl series."""

    p = pnl.dropna()
    if len(p) < 30 or p.std() == 0:
        return np.nan
    bpy = bars_per_year(p.index)
    if not np.isfinite(bpy):
        return np.nan
    return float(p.mean() / p.std() * math.sqrt(bpy))


def sortino(pnl):
    """Annualized Sortino ratio using downside deviation only."""

    p = pnl.dropna()
    if len(p) < 30:
        return np.nan
    bpy = bars_per_year(p.index)
    if not np.isfinite(bpy):
        return np.nan
    downside = np.minimum(p, 0)
    downside_dev = math.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return float(p.mean() / downside_dev * math.sqrt(bpy))


def cagr(pnl, capital):
    """Compound annual growth rate as a fraction of starting capital."""

    p = pnl.dropna()
    years = (p.index.max() - p.index.min()).total_seconds() / (365.25 * 86400)
    total = p.sum()
    if years <= 0 or total <= -capital:
        return np.nan
    return (1 + total / capital) ** (1 / years) - 1


def max_drawdown(pnl, capital):
    """Largest peak-to-trough loss as a fraction of starting capital."""

    cumulative = pnl.dropna().cumsum()
    if len(cumulative) == 0:
        return np.nan
    return float((cumulative - cumulative.cummax()).min() / capital)


def calmar(pnl, capital):
    """Compound annual growth rate divided by absolute max drawdown."""

    growth = cagr(pnl, capital)
    drawdown = max_drawdown(pnl, capital)
    if not np.isfinite(growth) or not np.isfinite(drawdown) or drawdown == 0:
        return np.nan
    return float(growth / abs(drawdown))
