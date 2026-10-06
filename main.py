"""Run the offline demo and print the three-strategy evaluation."""

from __future__ import annotations

from eval import evaluate, format_summary


def main() -> None:
    evaluation = evaluate()
    print("Flight-booking agent evaluation (mock data; no API key required)\n")
    print(format_summary(evaluation))
    failed = [row for row in evaluation["rows"] if not row["correct_outcome"]]
    print(f"\nCases: {len(evaluation['rows'])}; failed expected outcomes: {len(failed)}")
    if failed:
        for row in failed:
            print(f"- {row['agent']} / {row['case']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()