"""Three deterministic agent strategies for the flight-booking mock service."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from harness import FlightRequest, HarnessContext


class AgentState(TypedDict):
    request: FlightRequest
    context: HarnessContext
    actions: list[str]
    index: int
    memory: dict[str, Any]
    result: dict[str, Any] | None
    done: bool


def _validate_request(request: FlightRequest) -> str | None:
    if request.intent == "search":
        if not request.origin or not request.destination or not request.date:
            return "Search needs origin, destination, and date."
    elif request.intent == "details":
        if not request.flight_id:
            return "Flight details need a flight_id."
    elif request.intent == "book":
        if not request.confirm_booking:
            return "Booking was not explicitly confirmed; handing off without booking."
        if not request.passenger_name:
            return "Booking needs a passenger name."
        if request.passengers < 1:
            return "Booking needs at least one passenger."
        if not request.flight_id and not (
            request.origin and request.destination and request.date
        ):
            return "Booking needs a flight_id or complete search criteria."
    elif request.intent == "cancel":
        if not request.booking_id:
            return "Cancellation needs a booking_id."
    else:
        return f"Unsupported request intent: {request.intent}."
    return None


def _plan_actions(request: FlightRequest) -> list[str]:
    if request.intent == "search":
        return ["search"]
    if request.intent == "details":
        return ["details"]
    if request.intent == "cancel":
        return ["cancel"]
    if request.intent == "book":
        actions = [] if request.flight_id else ["search"]
        actions.extend(["details", "book"])
        return actions
    return []


def _execute_action(
    action: str,
    request: FlightRequest,
    context: HarnessContext,
    memory: dict[str, Any],
) -> dict[str, Any]:
    if action == "search":
        if not request.origin or not request.destination or not request.date:
            return {"ok": False, "error": "Complete search criteria are required."}
        result = context.call_tool(
            "search_flights",
            origin=request.origin,
            destination=request.destination,
            date=request.date,
            max_price=request.max_price,
            passengers=request.passengers,
        )
        if result.get("ok") and request.intent == "book" and not request.flight_id:
            flights = result["flights"]
            if not flights:
                return {"ok": False, "error": "No available flights match the request."}
            memory["flight_id"] = flights[0]["flight_id"]
            context.record_event(
                "decision",
                choice="lowest_price_flight",
                flight_id=memory["flight_id"],
            )
        return result
    if action == "details":
        flight_id = memory.get("flight_id") or request.flight_id
        if not flight_id:
            return {"ok": False, "error": "A flight_id is required to get details."}
        result = context.call_tool("get_flight_details", flight_id=flight_id)
        if result.get("ok"):
            memory["flight_id"] = result["flight"]["flight_id"]
            memory["details_checked"] = True
        return result
    if action == "book":
        flight_id = memory.get("flight_id") or request.flight_id
        if not flight_id or not request.passenger_name:
            return {"ok": False, "error": "Flight and passenger details are required."}
        return context.call_tool(
            "book_flight",
            flight_id=flight_id,
            passenger_name=request.passenger_name,
            passenger_email=context.user_email,
            passengers=request.passengers,
        )
    if action == "cancel":
        if not request.booking_id:
            return {"ok": False, "error": "A booking_id is required to cancel."}
        return context.call_tool("cancel_booking", booking_id=request.booking_id)
    return {"ok": False, "error": f"Unknown planned action: {action}."}


def _response(
    name: str,
    context: HarnessContext,
    result: dict[str, Any] | None,
    handoff: bool,
    message: str,
) -> dict[str, Any]:
    return {
        "strategy": name,
        "result": result,
        "handoff": handoff,
        "message": message,
        "trace": list(context.events),
    }


class ReActAgent:
    """Selects an action, observes its result, then decides the next action."""

    name = "ReAct"

    def run(self, request: FlightRequest, context: HarnessContext) -> dict[str, Any]:
        problem = _validate_request(request)
        if problem:
            context.record_event("handoff", reason=problem)
            return _response(self.name, context, None, True, problem)

        memory: dict[str, Any] = {}
        result: dict[str, Any] | None = None
        remaining = 4
        while remaining:
            if request.intent == "book":
                if not request.flight_id and "flight_id" not in memory:
                    action = "search"
                elif not memory.get("details_checked"):
                    action = "details"
                else:
                    action = "book"
            else:
                action = _plan_actions(request)[0]
                if result is not None:
                    break
            context.record_event("decision", strategy=self.name, action=action)
            result = _execute_action(action, request, context, memory)
            remaining -= 1
            if not result.get("ok"):
                context.record_event("handoff", reason=result.get("error", "Tool failed."))
                return _response(
                    self.name,
                    context,
                    result,
                    True,
                    result.get("error", "The tool call failed."),
                )
            if request.intent != "book" or action == "book":
                break
        if result is None:
            message = "No action was selected."
            return _response(self.name, context, None, True, message)
        return _response(self.name, context, result, False, "Request processed.")


class PlanThenExecuteAgent:
    """Creates a fixed, inspectable action plan before executing it."""

    name = "Plan-then-Execute"

    def run(self, request: FlightRequest, context: HarnessContext) -> dict[str, Any]:
        problem = _validate_request(request)
        if problem:
            context.record_event("handoff", reason=problem)
            return _response(self.name, context, None, True, problem)

        actions = _plan_actions(request)
        context.record_event("plan", strategy=self.name, actions=list(actions))
        memory: dict[str, Any] = {}
        result: dict[str, Any] | None = None
        for action in actions:
            context.record_event("execution_step", strategy=self.name, action=action)
            result = _execute_action(action, request, context, memory)
            if not result.get("ok"):
                context.record_event("handoff", reason=result.get("error", "Tool failed."))
                return _response(
                    self.name,
                    context,
                    result,
                    True,
                    result.get("error", "The tool call failed."),
                )
        if result is None:
            message = "The plan contained no executable actions."
            return _response(self.name, context, None, True, message)
        return _response(self.name, context, result, False, "Plan executed.")


class HybridAgent:
    """Plans with LangGraph, then executes graph steps until done or blocked."""

    name = "Hybrid (LangGraph)"

    def __init__(self) -> None:
        graph = StateGraph(AgentState)
        graph.add_node("make_plan", self._make_plan)
        graph.add_node("execute_step", self._execute_step)
        graph.add_edge(START, "make_plan")
        graph.add_edge("make_plan", "execute_step")
        graph.add_conditional_edges(
            "execute_step",
            self._route_after_step,
            {"execute_step": "execute_step", END: END},
        )
        self._graph = graph.compile()

    @staticmethod
    def _make_plan(state: AgentState) -> dict[str, Any]:
        actions = _plan_actions(state["request"])
        state["context"].record_event(
            "plan", strategy="Hybrid (LangGraph)", actions=list(actions)
        )
        return {"actions": actions, "index": 0, "memory": {}, "done": not actions}

    @staticmethod
    def _execute_step(state: AgentState) -> dict[str, Any]:
        action = state["actions"][state["index"]]
        context = state["context"]
        context.record_event(
            "execution_step", strategy="Hybrid (LangGraph)", action=action
        )
        result = _execute_action(action, state["request"], context, state["memory"])
        done = not result.get("ok") or state["index"] + 1 >= len(state["actions"])
        return {
            "result": result,
            "index": state["index"] + 1,
            "done": done,
        }

    @staticmethod
    def _route_after_step(state: AgentState) -> str:
        return END if state["done"] else "execute_step"

    def run(self, request: FlightRequest, context: HarnessContext) -> dict[str, Any]:
        problem = _validate_request(request)
        if problem:
            context.record_event("handoff", reason=problem)
            return _response(self.name, context, None, True, problem)

        final_state = self._graph.invoke(
            {
                "request": request,
                "context": context,
                "actions": [],
                "index": 0,
                "memory": {},
                "result": None,
                "done": False,
            }
        )
        result = final_state["result"]
        if result is None or not result.get("ok"):
            message = (
                result.get("error", "The tool call failed.")
                if result
                else "The graph completed without a result."
            )
            context.record_event("handoff", reason=message)
            return _response(self.name, context, result, True, message)
        return _response(self.name, context, result, False, "Graph plan executed.")