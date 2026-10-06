"""Mock flight-booking tools exposed to the agent and LangChain."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any

from langchain_core.tools import StructuredTool


class FlightToolError(Exception):
    """Expected, user-visible failure raised by a flight tool."""

    code = "flight_tool_error"


class InvalidInputError(FlightToolError):
    code = "invalid_input"


class FlightNotFoundError(FlightToolError):
    code = "flight_not_found"


class BookingNotFoundError(FlightToolError):
    code = "booking_not_found"


class BookingStateError(FlightToolError):
    code = "booking_state_error"


_FLIGHTS: list[dict[str, Any]] = [
    {
        "flight_id": "VN101",
        "airline": "Vietnam Airlines",
        "origin": "HAN",
        "destination": "SGN",
        "date": "2026-11-12",
        "depart_time": "08:00",
        "arrive_time": "10:10",
        "price": 1_850_000,
        "currency": "VND",
        "seats_available": 12,
    },
    {
        "flight_id": "VJ203",
        "airline": "VietJet Air",
        "origin": "HAN",
        "destination": "SGN",
        "date": "2026-11-12",
        "depart_time": "10:30",
        "arrive_time": "12:40",
        "price": 1_290_000,
        "currency": "VND",
        "seats_available": 8,
    },
    {
        "flight_id": "QH205",
        "airline": "Bamboo Airways",
        "origin": "HAN",
        "destination": "DAD",
        "date": "2026-11-12",
        "depart_time": "13:15",
        "arrive_time": "14:35",
        "price": 1_420_000,
        "currency": "VND",
        "seats_available": 5,
    },
    {
        "flight_id": "VN102",
        "airline": "Vietnam Airlines",
        "origin": "SGN",
        "destination": "HAN",
        "date": "2026-11-13",
        "depart_time": "17:00",
        "arrive_time": "19:10",
        "price": 1_920_000,
        "currency": "VND",
        "seats_available": 10,
    },
]

_bookings: dict[str, dict[str, Any]] = {}
_booking_sequence = 1000
_data_lock = RLock()


def search_flights(
    origin: str,
    destination: str,
    date: str,
    max_price: int | None = None,
    passengers: int = 1,
) -> dict[str, Any]:
    """Search mock flights by airport codes, travel date, and optional limits."""
    if not all(isinstance(value, str) for value in (origin, destination, date)):
        raise InvalidInputError("origin, destination, and date must be strings.")
    if not isinstance(passengers, int) or isinstance(passengers, bool):
        raise InvalidInputError("passengers must be an integer.")
    if max_price is not None and (
        not isinstance(max_price, int) or isinstance(max_price, bool)
    ):
        raise InvalidInputError("max_price must be an integer.")
    origin_code = origin.strip().upper()
    destination_code = destination.strip().upper()
    if not origin_code or not destination_code or not date.strip():
        raise InvalidInputError("origin, destination, and date are required.")
    if origin_code == destination_code:
        raise InvalidInputError("origin and destination must be different.")
    if passengers < 1:
        raise InvalidInputError("passengers must be at least 1.")
    if max_price is not None and max_price < 0:
        raise InvalidInputError("max_price cannot be negative.")

    with _data_lock:
        matches = [
            deepcopy(flight)
            for flight in _FLIGHTS
            if flight["origin"] == origin_code
            and flight["destination"] == destination_code
            and flight["date"] == date.strip()
            and flight["seats_available"] >= passengers
            and (max_price is None or flight["price"] <= max_price)
        ]
    matches.sort(key=lambda flight: (flight["price"], flight["depart_time"]))
    return {"ok": True, "flights": matches, "count": len(matches)}


def get_flight_details(flight_id: str) -> dict[str, Any]:
    """Return details for one mock flight."""
    if not isinstance(flight_id, str):
        raise InvalidInputError("flight_id must be a string.")
    requested_id = flight_id.strip().upper()
    if not requested_id:
        raise InvalidInputError("flight_id is required.")
    with _data_lock:
        for flight in _FLIGHTS:
            if flight["flight_id"] == requested_id:
                return {"ok": True, "flight": deepcopy(flight)}
    raise FlightNotFoundError(f"No flight found for flight_id={requested_id}.")


def book_flight(
    flight_id: str,
    passenger_name: str,
    passenger_email: str,
    passengers: int = 1,
) -> dict[str, Any]:
    """Create a mock booking if the flight has sufficient available seats."""
    if not all(isinstance(value, str) for value in (flight_id, passenger_name, passenger_email)):
        raise InvalidInputError("flight_id, passenger_name, and passenger_email must be strings.")
    if not isinstance(passengers, int) or isinstance(passengers, bool):
        raise InvalidInputError("passengers must be an integer.")
    requested_id = flight_id.strip().upper()
    name = passenger_name.strip()
    email = passenger_email.strip().lower()
    if not requested_id or not name or not email:
        raise InvalidInputError("flight_id, passenger_name, and passenger_email are required.")
    if "@" not in email:
        raise InvalidInputError("passenger_email must be a valid email address.")
    if passengers < 1:
        raise InvalidInputError("passengers must be at least 1.")

    global _booking_sequence
    with _data_lock:
        flight = next((item for item in _FLIGHTS if item["flight_id"] == requested_id), None)
        if flight is None:
            raise FlightNotFoundError(f"No flight found for flight_id={requested_id}.")
        if flight["seats_available"] < passengers:
            raise BookingStateError("Not enough seats are available for this flight.")

        booking_id = f"BKG{_booking_sequence}"
        _booking_sequence += 1
        booking = {
            "booking_id": booking_id,
            "flight_id": requested_id,
            "passenger_name": name,
            "passenger_email": email,
            "passengers": passengers,
            "status": "confirmed",
        }
        flight["seats_available"] -= passengers
        _bookings[booking_id] = booking
        return {"ok": True, "booking": deepcopy(booking)}


def cancel_booking(booking_id: str, requester_email: str) -> dict[str, Any]:
    """Cancel a booking only when it belongs to the requesting customer."""
    if not isinstance(booking_id, str) or not isinstance(requester_email, str):
        raise InvalidInputError("booking_id and requester_email must be strings.")
    requested_id = booking_id.strip().upper()
    email = requester_email.strip().lower()
    if not requested_id or not email:
        raise InvalidInputError("booking_id and requester_email are required.")
    with _data_lock:
        booking = _bookings.get(requested_id)
        if booking is None:
            raise BookingNotFoundError(f"No booking found for booking_id={requested_id}.")
        if booking["passenger_email"] != email:
            raise BookingStateError("The booking does not belong to the requesting customer.")
        if booking["status"] == "cancelled":
            raise BookingStateError("This booking has already been cancelled.")
        booking["status"] = "cancelled"
        flight = next(item for item in _FLIGHTS if item["flight_id"] == booking["flight_id"])
        flight["seats_available"] += booking["passengers"]
        return {"ok": True, "booking": deepcopy(booking)}


def reset_mock_data() -> None:
    """Reset mutable mock bookings so demos and evaluations are repeatable."""
    global _booking_sequence
    with _data_lock:
        _bookings.clear()
        _booking_sequence = 1000
        initial_seats = {"VN101": 12, "VJ203": 8, "QH205": 5, "VN102": 10}
        for flight in _FLIGHTS:
            flight["seats_available"] = initial_seats[flight["flight_id"]]


def list_bookings() -> list[dict[str, Any]]:
    """Return a copy of all mock bookings; intended for demos and tests."""
    with _data_lock:
        return deepcopy(list(_bookings.values()))


LANGCHAIN_TOOLS = (
    StructuredTool.from_function(
        func=search_flights,
        name="search_flights",
        description="Find mock flights using origin/destination airport codes and a YYYY-MM-DD date.",
    ),
    StructuredTool.from_function(
        func=get_flight_details,
        name="get_flight_details",
        description="Get the mock schedule, airline, price, and seat count for a flight ID.",
    ),
    StructuredTool.from_function(
        func=book_flight,
        name="book_flight",
        description="Book available seats on a flight for a named passenger and email address.",
    ),
    StructuredTool.from_function(
        func=cancel_booking,
        name="cancel_booking",
        description="Cancel a booking after verifying the requester's email matches its owner.",
    ),
)