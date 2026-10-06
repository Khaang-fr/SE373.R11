"""A small, deterministic harness for permissioned flight-booking agents."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from tools import FlightToolError, book_flight, cancel_booking, get_flight_details, search_flights


@dataclass(frozen=True)
class FlightRequest:
    """Structured task input used by the agents and the reproducible evaluation."""

    intent: str
    origin: str | None = None
    destination: str | None = None
    date: str | None = None
    flight_id: str | None = None
    booking_id: str | None = None
    passenger_name: str | None = None
    passenger_email: str | None = None
    passengers: int = 1
    max_price: int | None = None
    confirm_booking: bool = False


@dataclass(frozen=True)
class HarnessPolicy:
    """Policy is data; HarnessContext enforces it at every tool boundary."""

    tool_permissions: dict[str, frozenset[str]] = field(
        default_factory=lambda: {
            "search_flights": frozenset({"guest", "customer", "admin"}),
            "get_flight_details": frozenset({"guest", "customer", "admin"}),
            "book_flight": frozenset({"customer", "admin"}),
            "cancel_booking": frozenset({"customer", "admin"}),
        }
    )
    constraints: dict[str, Any] = field(
        default_factory=lambda: {
            "max_tool_calls": 8,
            "max_passengers": 6,
            "allowed_travel_dates": ("2026-11-12", "2026-11-13"),
            "require_booking_confirmation": True,
        }
    )


class HarnessContext:
    """Authenticated identity, policy checks, and an auditable tool-call log."""

    _TOOL_FUNCTIONS = {
        "search_flights": search_flights,
        "get_flight_details": get_flight_details,
        "book_flight": book_flight,
        "cancel_booking": cancel_booking,
    }

    def __init__(
        self,
        request: FlightRequest,
        user_email: str,
        role: str,
        policy: HarnessPolicy,
    ) -> None:
        self.request = request
        self.user_email = user_email.strip().lower()
        self.role = role
        self.policy = policy
        self.tool_calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.policy_violations: list[str] = []

    def record_event(self, event_type: str, **details: Any) -> None:
        self.events.append({"type": event_type, **details})

    def call_tool(self, name: str, **arguments: Any) -> dict[str, Any]:
        """Run one allowlisted tool call, enforcing permissions and constraints."""
        if name not in self._TOOL_FUNCTIONS:
            return self._reject(name, "unknown_tool", f"Tool {name!r} is not allowlisted.")
        roles = self.policy.tool_permissions.get(name, frozenset())
        if self.role not in roles:
            return self._reject(
                name,
                "permission_denied",
                f"Role {self.role!r} is not allowed to use {name}.",
            )
        max_calls = self.policy.constraints["max_tool_calls"]
        if len(self.tool_calls) >= max_calls:
            return self._reject(
                name,
                "tool_call_limit",
                f"Tool-call limit ({max_calls}) has been reached.",
            )

        args = dict(arguments)
        if name in {"search_flights", "book_flight"}:
            passenger_count = args.get("passengers", 1)
            max_passengers = self.policy.constraints["max_passengers"]
            if not isinstance(passenger_count, int) or isinstance(passenger_count, bool):
                return self._reject(
                    name,
                    "invalid_passenger_count",
                    "passengers must be an integer.",
                )
            if passenger_count > max_passengers:
                return self._reject(
                    name,
                    "passenger_limit",
                    f"Policy allows at most {max_passengers} passengers.",
                )
        if name == "search_flights":
            allowed_dates = self.policy.constraints["allowed_travel_dates"]
            if args.get("date") not in allowed_dates:
                return self._reject(
                    name,
                    "date_not_allowed",
                    "The requested date is outside the mock service's allowed dates.",
                )
        if name == "book_flight":
            if (
                self.policy.constraints["require_booking_confirmation"]
                and not self.request.confirm_booking
            ):
                return self._reject(
                    name,
                    "confirmation_required",
                    "Explicit booking confirmation is required.",
                )
            try:
                flight = get_flight_details(args.get("flight_id", ""))["flight"]
            except FlightToolError as error:
                return self._reject(name, error.code, str(error))
            if flight["date"] not in self.policy.constraints["allowed_travel_dates"]:
                return self._reject(
                    name,
                    "date_not_allowed",
                    "The selected flight date is outside the allowed dates.",
                )
            args["passenger_email"] = self.user_email
        elif name == "cancel_booking":
            args["requester_email"] = self.user_email

        try:
            result = self._TOOL_FUNCTIONS[name](**args)
        except FlightToolError as error:
            result = {"ok": False, "error": str(error), "code": error.code}
        self.tool_calls.append(
            {"tool": name, "arguments": deepcopy(args), "result": deepcopy(result)}
        )
        self.record_event("tool_result", tool=name, ok=result["ok"])
        return result

    def _reject(self, name: str, code: str, message: str) -> dict[str, Any]:
        self.policy_violations.append(message)
        result = {"ok": False, "error": message, "code": code}
        self.tool_calls.append(
            {"tool": name, "arguments": {}, "result": deepcopy(result)}
        )
        self.record_event("policy_rejection", tool=name, code=code)
        return result


class Harness:
    """Runs an agent and evaluates its outcome against code-based criteria."""

    def __init__(self, policy: HarnessPolicy | None = None) -> None:
        self.policy = policy or HarnessPolicy()

    def run(
        self,
        agent: Any,
        request: FlightRequest,
        *,
        user_email: str = "linh@example.test",
        role: str = "customer",
    ) -> dict[str, Any]:
        context = HarnessContext(request, user_email, role, self.policy)
        agent_output = agent.run(request, context)
        completed, completion_reason = self._check_completion(request, context)
        return {
            "agent": agent.name,
            "intent": request.intent,
            "completed": completed,
            "completion_reason": completion_reason,
            "handoff": bool(agent_output.get("handoff", False)),
            "message": agent_output.get("message", ""),
            "result": deepcopy(agent_output.get("result")),
            "tool_calls": deepcopy(context.tool_calls),
            "tool_call_count": len(context.tool_calls),
            "policy_violations": list(context.policy_violations),
            "trace": deepcopy(context.events),
        }

    @staticmethod
    def _check_completion(
        request: FlightRequest,
        context: HarnessContext,
    ) -> tuple[bool, str]:
        expected_tool = {
            "search": "search_flights",
            "details": "get_flight_details",
            "book": "book_flight",
            "cancel": "cancel_booking",
        }.get(request.intent)
        if expected_tool is None:
            return False, "unsupported_intent"
        successful_calls = [
            call
            for call in context.tool_calls
            if call["tool"] == expected_tool and call["result"].get("ok")
        ]
        if successful_calls:
            return True, f"{expected_tool}_succeeded"
        if context.policy_violations:
            return False, "policy_blocked"
        return False, f"{expected_tool}_not_completed"