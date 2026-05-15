"""
Alpha report formatter — render research_brief.json as human-readable report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def format_alpha_report(report: Dict[str, Any], rich: bool = True) -> str:
    """Format a research report dict as a printable string."""
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"ALPHA RESEARCH REPORT  run_id={report.get('run_id', '?')}")
    lines.append(f"Topic: {report.get('topic', '?')}")
    lines.append(f"Generated: {report.get('generated_at', '?')}")
    lines.append(f"Universe: {', '.join(report.get('universe', []))}")
    lines.append(sep)

    # Macro
    macro = report.get("macro_regime", {})
    lines.append("\nMACRO REGIME")
    lines.append(f"  Overall: {macro.get('overall_regime', '?')} "
                 f"(confidence={macro.get('regime_confidence', 0):.0%})")
    lines.append(f"  Primary driver: {macro.get('primary_driver', '?')}")
    lines.append(f"  Summary: {macro.get('macro_summary', '?')}")

    # Sentiment
    sentiment = report.get("sentiment_summary", {})
    options = report.get("options_summary", {})
    lines.append("\nMARKET CONDITIONS")
    lines.append(f"  Fear/Greed: {sentiment.get('fear_greed', '?')}")
    lines.append(f"  VIX environment: {sentiment.get('vix_env', '?')}")
    lines.append(f"  Options smart money: {options.get('smart_money', '?')} "
                 f"(confidence={options.get('confidence', 0):.0%})")

    # Signals
    signals = report.get("alpha_signals", [])
    lines.append(f"\nALPHA SIGNALS ({len(signals)} found)")
    for s in sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True):
        lines.append(
            f"  [{s.get('confidence', 0):.0%}] {s.get('name')} "
            f"({s.get('ticker')}) — {s.get('direction')} {s.get('signal_type')}"
        )
        for ev in s.get("evidence", [])[:2]:
            lines.append(f"    • {ev}")

    # Backtest results
    results = report.get("strategy_results", [])
    lines.append(f"\nSTRATEGY BACKTEST RESULTS ({report.get('validated_count', 0)} passed)")
    for r in results[:5]:
        passed = "✓" if r.get("backtest_passed") else "✗"
        lines.append(
            f"  {passed} {r.get('signal_name')} ({r.get('ticker_tested', '?')}): "
            f"val_sharpe={r.get('val_sharpe', 0):.3f} "
            f"train={r.get('train_sharpe', 0):.3f} "
            f"oos={r.get('oos_ratio', 0):.2f} "
            f"trades={r.get('total_trades', 0)}"
        )

    top = report.get("top_strategy")
    if top and top.get("backtest_passed"):
        lines.append(f"\nTOP STRATEGY: {top.get('signal_name')}")
        lines.append(f"  Val Sharpe: {top.get('val_sharpe', 0):.4f}")
        lines.append(f"  OOS Ratio: {top.get('oos_ratio', 0):.2f}")
        lines.append(f"  Best strategy saved to: artifacts/strategy_candidate.py")
    elif results:
        lines.append(f"\nNo strategies passed full validation. Best attempt:")
        lines.append(f"  {results[0].get('signal_name')}: val_sharpe={results[0].get('val_sharpe', 0):.3f}")

    lines.append(f"\nResearch brief: artifacts/research_brief.json")
    lines.append(sep)

    return "\n".join(lines)


def load_brief(project_root: Path) -> Dict[str, Any]:
    """Load the most recent research brief."""
    brief_path = project_root / "artifacts" / "research_brief.json"
    if not brief_path.exists():
        return {}
    return json.loads(brief_path.read_text())


def brief_to_context_string(brief: Dict[str, Any], max_chars: int = 3000) -> str:
    """
    Convert a research brief to a compact string for injecting into agent prompts.
    """
    if not brief:
        return ""

    parts = [
        f"## Research Brief (run_id={brief.get('run_id', '?')})",
        f"Topic: {brief.get('topic', '?')}",
        f"Macro: {brief.get('macro_regime', {}).get('overall_regime', 'unknown')} regime",
        f"Sentiment: {brief.get('sentiment_summary', {}).get('fear_greed', 'unknown')}",
    ]

    signals = brief.get("alpha_signals", [])
    if signals:
        parts.append(f"\nTop alpha signals:")
        for s in sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)[:3]:
            parts.append(
                f"  - {s.get('name')} ({s.get('ticker')}) "
                f"[{s.get('direction')} {s.get('signal_type')}, confidence={s.get('confidence', 0):.0%}]"
            )
            for ev in s.get("evidence", [])[:1]:
                parts.append(f"    Evidence: {ev}")
            parts.append(f"    Entry: {s.get('suggested_entry', '?')}")

    top = brief.get("top_strategy")
    if top and top.get("backtest_passed"):
        parts.append(
            f"\nBest validated strategy: {top.get('signal_name')} "
            f"(val_sharpe={top.get('val_sharpe', 0):.3f})"
        )

    result = "\n".join(parts)
    return result[:max_chars]
