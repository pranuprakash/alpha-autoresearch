"""Format portfolio + recommendation report as a terminal-readable string."""

from __future__ import annotations

from .models import Portfolio, Position
from .recommender import RecommendationReport


def format_portfolio_report(portfolio: Portfolio, report: RecommendationReport) -> str:
    lines = []
    s = report.portfolio_summary

    lines += [
        "=" * 65,
        "  PORTFOLIO SUMMARY",
        "=" * 65,
        f"  Total Value:     ${s['total_value']:>12,.2f}",
        f"  Total P&L:       ${s['total_pnl']:>+12,.2f}  ({s['total_pnl_pct']:+.1f}%)",
        f"  Cash:            ${s['cash']:>12,.2f}",
        f"  Positions:       {s['position_count']}",
        f"  Net Delta:       {s['net_delta']:>+12.1f}",
        f"  Net Theta:       {s['net_theta']:>+12.2f} /day",
        f"  Net Vega:        {s['net_vega']:>+12.2f} per 1% IV",
        f"  1-Day VaR 95%:   ${s['var_95_1d']:>12,.0f}",
        "",
    ]

    if report.risk_notes:
        lines.append("  RISK NOTES")
        for note in report.risk_notes:
            lines.append(f"  ⚠  {note}")
        lines.append("")

    # Positions detail
    lines += ["=" * 65, "  CURRENT POSITIONS", "=" * 65]
    for pos in portfolio.positions:
        label = pos.option_ticker or pos.symbol
        val = pos.current_value or 0
        pnl = pos.pnl or 0
        pnl_pct = pos.pnl_pct or 0
        lines.append(f"  {label:<30} ${val:>10,.2f}  P&L: ${pnl:>+8,.2f} ({pnl_pct:+.1f}%)")
        if pos.asset_type == "option":
            lines.append(
                f"    DTE: {pos.dte}  δ={pos.delta:.2f}  θ={pos.theta:.3f}/day  "
                f"IV: {pos.iv:.0f}%"
            )
    lines.append("")

    # Position actions
    if report.position_actions:
        lines += ["=" * 65, "  POSITION ACTIONS", "=" * 65]
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        for action in sorted(report.position_actions,
                              key=lambda a: urgency_order.get(a.urgency, 3)):
            marker = "!!" if action.urgency == "HIGH" else " >" if action.urgency == "MEDIUM" else "  "
            lines.append(f"  {marker} [{action.action:<5}] {action.symbol}")
            lines.append(f"         {action.reason}")
            if action.details:
                lines.append(f"         {action.details}")
        lines.append("")

    # New plays
    if report.new_plays:
        lines += ["=" * 65, "  NEW TRADE TICKETS", "=" * 65]
        for i, play in enumerate(report.new_plays, 1):
            conf = play.get("confidence", 0)
            conf_label = "HIGH" if conf > 0.7 else "MEDIUM" if conf > 0.5 else "LOW"
            inst = play.get("instrument") or f"{play.get('ticker')} equity"
            qty = play.get("quantity") or play.get("shares", 1)

            lines.append("")
            lines.append(f"  PRIORITY {i} — {conf_label} CONFIDENCE")
            lines.append(f"  {play.get('action')} {qty}x {inst}")
            lines.append(f"  Entry: ${play.get('entry_price'):.2f}  Limit: ${play.get('entry_limit'):.2f}")
            lines.append(
                f"  Target: ${play.get('target_price'):.2f}  "
                f"Stop: ${play.get('stop_price'):.2f}  "
                f"R/R: {play.get('risk_reward', 0):.1f}:1"
            )
            cap = play.get("capital_at_risk", 0)
            pct = play.get("portfolio_pct", 0)
            lines.append(f"  Risk: ${cap:,.0f} ({pct:.1f}% portfolio)")
            if "iv_rank" in play:
                lines.append(
                    f"  IV Rank: {play['iv_rank']:.0f}th pct  "
                    f"Delta: {play.get('delta', 0):.2f}"
                )
            if play.get("rationale"):
                lines.append(f"  Rationale: {play['rationale'][:120]}")
        lines.append("")

    # Capital summary
    cap = report.capital_summary
    lines += [
        "=" * 65,
        "  CAPITAL SUMMARY",
        "=" * 65,
        f"  Cash available:        ${cap['cash_available']:>10,.2f}",
        f"  Committed to plays:    ${cap['committed_to_new_plays']:>10,.2f}",
        f"  Remaining after plays: ${cap['remaining_after_plays']:>10,.2f}",
        "=" * 65,
    ]

    return "\n".join(lines)
