"""Render evidence/RESULTS.md from the JSON the jobs write.

    python scripts/report.py

The README's results are copied from this file, so every number in the
README traces back to a measured value in evidence/*.json.
"""

from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def segment_titles(m: dict) -> dict[str, str]:
    first_test = m["split"]["boundaries"]["valid_end_step"] + 1
    normal_end = m["assumptions"]["normal_volume_last_step"]
    return {
        "test_normal_volume": f"Normal-volume test hours (steps {first_test}-{normal_end})",
        "test_low_volume_tail": f"Low-volume tail (steps after {normal_end})",
        "test_all": f"Whole test period (steps from {first_test})",
    }


NAMES = {"xgboost": "XGBoost", "logistic_regression": "Logistic regression"}
CHECKS = {
    "leakage_check": "XGBoost + balance columns (check, not deployable)",
    "without_time_of_day_check": "XGBoost without time of day (check, not deployable)",
}


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def ci(d: dict) -> str:
    return f"{d['estimate']:.3f} ({d['ci_low']:.3f}-{d['ci_high']:.3f})"


def main() -> None:
    m = json.loads((EVIDENCE / "metrics.json").read_text())
    p = json.loads((EVIDENCE / "data_profile.json").read_text())
    out: list[str] = ["# Results", ""]
    out.append(f"Generated from `evidence/metrics.json` (run time {m['runtime_seconds']} s).")
    out.append(f"Input file SHA-256: `{p['source_sha256']}`")
    out += ["", "## Data", ""]
    out.append(f"- Rows: {p['rows']:,}; fraud: {p['fraud_rows']:,} ({pct(p['fraud_rate'], 3)})")
    split = m["split"]
    b = split["boundaries"]
    out.append(
        f"- Split by step: train <= {b['train_end_step']}, "
        f"validation <= {b['valid_end_step']}, test after"
    )
    out += ["", "| Split | Rows | Fraud | Fraud rate |", "|---|---:|---:|---:|"]
    for name in ("train", "valid", "test"):
        s = split[name]
        out.append(f"| {name} | {s['rows']:,} | {s['fraud']:,} | {pct(s['fraud_rate'], 3)} |")

    out += ["", f"Champion (chosen on validation): **{NAMES[m['champion']]}**", ""]
    out += ["| Model | Validation AP |", "|---|---:|"]
    for key, r in m["models"].items():
        out.append(f"| {NAMES[key]} | {r['valid_average_precision']:.3f} |")

    for seg, title in segment_titles(m).items():
        first = next(iter(m["models"].values())).get(seg)
        if first is None:  # segment empty for this dataset
            continue
        out += ["", f"## {title}", ""]
        out.append(
            f"{first['rows']:,} transactions, {first['fraud']:,} fraud "
            f"({pct(first['fraud_rate'], 3)})."
        )
        out += [
            "",
            "| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall "
            "| Value recall |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for key, r in m["models"].items():
            for i, a in enumerate(r[seg]["at_budget"]):
                ap = ci(r[seg]["average_precision"]) if i == 0 else ""
                name = NAMES[key] if i == 0 else ""
                out.append(
                    f"| {name} | {ap} | top {pct(a['budget'])} | {a['alerts']:,} | "
                    f"{pct(a['precision'])} | {pct(a['recall'])} | {pct(a['value_recall'])} |"
                )
        for check_key, label in CHECKS.items():
            check = m.get(check_key, {}).get(seg)
            if check:
                for i, a in enumerate(check["at_budget"]):
                    ap = f"{check['average_precision']:.3f}" if i == 0 else ""
                    name = f"*{label}*" if i == 0 else ""
                    out.append(
                        f"| {name} | {ap} | top {pct(a['budget'])} | {a['alerts']:,} | "
                        f"{pct(a['precision'])} | {pct(a['recall'])} | {pct(a['value_recall'])} |"
                    )

        c = m["xgboost_minus_logistic_average_precision"][seg]
        out += [
            "",
            f"XGBoost minus logistic regression, average precision: {c['difference']:+.3f} "
            f"(95% CI {c['ci_low']:+.3f} to {c['ci_high']:+.3f}; "
            f"share of resamples where XGBoost was not better: {c['share_b_not_better']:.3f}).",
        ]

        rule = m["existing_rule_on_test"][seg]
        out += ["", "**Existing rule vs the model at the same alert count**", ""]
        out += ["| | Alerts | Precision | Recall | Value recall |", "|---|---:|---:|---:|---:|"]
        out.append(
            f"| `isFlaggedFraud` rule | {rule['alerts']:,} | {pct(rule['precision'])} | "
            f"{pct(rule['recall'])} | {pct(rule['value_recall'])} |"
        )
        champ = m["models"][m["champion"]][seg]
        if "at_rule_volume" in champ:
            a = champ["at_rule_volume"]
            out.append(
                f"| {NAMES[m['champion']]} | {a['alerts']:,} | {pct(a['precision'])} | "
                f"{pct(a['recall'])} | {pct(a['value_recall'])} |"
            )

    out += ["", "## Cost analysis: computed, deliberately not used", ""]
    out.append(
        "`metrics.json` holds the cost-minimising alert volume for review costs of "
        f"{', '.join(f'{c:,.0f}' for c in m['assumptions']['review_costs'])} per alert. "
        "With PaySim's large transaction amounts, every one of those costs makes "
        "reviewing a large share of all transactions look optimal, which no fraud team could "
        "staff. Without real review costs and capacity, the cost model would mislead, so the "
        "results above are reported at fixed alert budgets instead."
    )

    mon = m.get("monitoring")
    if mon:
        out += ["", "## Operating point stored with the model", ""]
        out.append(
            f"- Alert threshold for the top {pct(mon['alert_budget'])} of transactions, set on "
            f"validation scores: {mon['alert_threshold']:.6f}"
        )
        out.append(
            f"- Score reference for drift monitoring: {len(mon['score_reference']['proportions'])} "
            "quantile bins of the validation scores"
        )

    out += ["", "## Export parity", ""]
    for key, v in m["export_parity_max_abs_diff"].items():
        out.append(f"- {NAMES[key]}: max |exported - Spark| = {v:.2e}")
    (EVIDENCE / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {EVIDENCE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
