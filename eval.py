"""Deterministic evaluation of the three flight-booking agent strategies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agents import HybridAgent, PlanThenExecuteAgent, ReActAgent
from harness import FlightRequest, Harness
from tools import book_flight, reset_mock_data


USER_EMAIL = "linh@example.test"


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    request: FlightRequest
    expected: str
    role: str = "customer"
    seed_booking: bool = False
    booking_owner_email: str = USER_EMAIL


CASES = (
    EvaluationCase(
        "search_available_flights",
        FlightRequest("search", "HAN", "SGN", "2026-11-12"),
        "completed",
    ),
    EvaluationCase(
        "get_flight_details",
        FlightRequest("details", flight_id="VN101"),
        "completed",
    ),
    EvaluationCase(
        "book_after_confirmation",
        FlightRequest(
            "book",
            flight_id="VJ203",
            passenger_name="Linh Nguyen",
            confirm_booking=True,
        ),
        "completed",
    ),
    EvaluationCase(
        "search_then_book_after_confirmation",
        FlightRequest(
            "book",
            origin="HAN",
            destination="SGN",
            date="2026-11-12",
            passenger_name="Linh Nguyen",
            confirm_booking=True,
        ),
        "completed",
    ),
    EvaluationCase(
        "cancel_owned_booking",
        FlightRequest("cancel", booking_id="BKG1000"),
        "completed",
        seed_booking=True,
    ),
    EvaluationCase(
        "handoff_unconfirmed_booking",
        FlightRequest(
            "book",
            flight_id="VN101",
            passenger_name="Linh Nguyen",
            confirm_booking=False,
        ),
        "handoff",
    ),
    EvaluationCase(
        "handoff_unsupported_refund",
        FlightRequest("refund", booking_id="BKG1000"),
        "handoff",
    ),
    EvaluationCase(
        "deny_guest_booking",
        FlightRequest(
            "book",
            flight_id="VN101",
            passenger_name="Linh Nguyen",
            confirm_booking=True,
        ),
        "handoff",
        role="guest",
    ),
    EvaluationCase(
        "block_disallowed_travel_date",
        FlightRequest("search", "HAN", "SGN", "2026-12-01"),
        "handoff",
    ),
    EvaluationCase(
        "deny_cancelling_another_customers_booking",
        FlightRequest("cancel", booking_id="BKG1000"),
        "handoff",
        seed_booking=True,
        booking_owner_email="another-customer@example.test",
    ),
)


def _seed_booking(owner_email: str) -> None:
    seeded = book_flight(
        flight_id="VN101",
        passenger_name="Fixture Passenger",
        passenger_email=owner_email,
    )
    if not seeded["ok"] or seeded["booking"]["booking_id"] != "BKG1000":
        raise RuntimeError("Failed to initialize the deterministic cancellation fixture.")


def evaluate() -> dict[str, Any]:
    """Run every strategy against every case and return raw rows and summaries."""
    agents = (ReActAgent(), PlanThenExecuteAgent(), HybridAgent())
    harness = Harness()
    rows: list[dict[str, Any]] = []

    for agent in agents:
        for case in CASES:
            reset_mock_data()
            if case.seed_booking:
                _seed_booking(case.booking_owner_email)
            started = perf_counter()
            result = harness.run(
                agent,
                case.request,
                user_email=USER_EMAIL,
                role=case.role,
            )
            elapsed_ms = (perf_counter() - started) * 1000
            if case.expected == "completed":
                correct = result["completed"] and not result["handoff"]
            else:
                correct = result["handoff"] and not result["completed"]
            rows.append(
                {
                    "agent": agent.name,
                    "case": case.name,
                    "expected": case.expected,
                    "correct_outcome": correct,
                    "completed": result["completed"],
                    "handoff": result["handoff"],
                    "tool_calls": result["tool_call_count"],
                    "trace_events": len(result["trace"]),
                    "policy_violations": len(result["policy_violations"]),
                    "elapsed_ms": elapsed_ms,
                }
            )

    summaries: list[dict[str, Any]] = []
    for agent in agents:
        agent_rows = [row for row in rows if row["agent"] == agent.name]
        summaries.append(
            {
                "agent": agent.name,
                "cases": len(agent_rows),
                "correct_outcomes": sum(row["correct_outcome"] for row in agent_rows),
                "outcome_accuracy": sum(row["correct_outcome"] for row in agent_rows)
                / len(agent_rows),
                "completed_tasks": sum(row["completed"] for row in agent_rows),
                "correct_handoffs": sum(
                    row["handoff"] and row["expected"] == "handoff"
                    for row in agent_rows
                ),
                "policy_blocks": sum(
                    row["policy_violations"] > 0 for row in agent_rows
                ),
                "mean_tool_calls": sum(row["tool_calls"] for row in agent_rows)
                / len(agent_rows),
                "mean_trace_events": sum(row["trace_events"] for row in agent_rows)
                / len(agent_rows),
                "mean_elapsed_ms": sum(row["elapsed_ms"] for row in agent_rows)
                / len(agent_rows),
            }
        )
    return {"cases": [case.name for case in CASES], "rows": rows, "summaries": summaries}


def format_summary(evaluation: dict[str, Any]) -> str:
    """Format compact, copy-ready metrics for the report."""
    lines = [
        "| Strategy | Correct outcomes | Accuracy | Correct handoffs | Policy blocks | "
        "Avg tool calls | Avg trace events | Avg time (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in evaluation["summaries"]:
        lines.append(
            f"| {summary['agent']} | {summary['correct_outcomes']}/{summary['cases']} "
            f"| {summary['outcome_accuracy']:.0%} | {summary['correct_handoffs']} "
            f"| {summary['policy_blocks']} | {summary['mean_tool_calls']:.2f} "
            f"| {summary['mean_trace_events']:.2f} | {summary['mean_elapsed_ms']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    report = evaluate()
    print(format_summary(report))
    for row in report["rows"]:
        print(
            f"{row['agent']} / {row['case']}: "
            f"{'PASS' if row['correct_outcome'] else 'FAIL'}, "
            f"tools={row['tool_calls']}, policy_blocks={row['policy_violations']}"
        )