import logging
import json
from datetime import datetime, timedelta
import re

from amadeus import Client, ResponseError

# -----------------------------
# 1. Configure Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# 2. Amadeus Credentials
# -----------------------------
amadeus_client = Client(
    client_id='pNYUh6kEI8K4JbqkXl79xCZcI2IMbs37',
    client_secret='v8hJNTTa1DfLvXG7'
)

# -----------------------------
# 3. Initialize Amadeus Client


# -----------------------------
# 4. Fetch Flight Offers
# -----------------------------
def get_flight_offers(amadeus_client, origin, destination,
                      departure_date, return_date,
                      adults=1, max_results=10):
    """
    Retrieves flight offers from Amadeus for round-trip flights.
    """
    try:
        response = amadeus_client.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            returnDate=return_date,
            adults=adults,
            max=max_results
        )
        return response.data  # List of raw flight offers (JSON format)
    except ResponseError as error:
        logging.error(f"Error while fetching flight offers: {error}")
        return []

# -----------------------------
# 5. Helper Parsing Functions
# -----------------------------
def iso_to_datetime(iso_str):
    """Convert ISO 8601 string (e.g. '2025-06-01T10:15:00') into a datetime object."""
    return datetime.fromisoformat(iso_str)

def parse_ISO8601_duration(duration_str):
    """
    Parse an ISO 8601 duration string like 'PT14H20M' into a timedelta object.
    (E.g., 'PT2H10M' => 2 hours, 10 minutes)
    """
    pattern = r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
    match = re.match(pattern, duration_str)
    if not match:
        return timedelta()
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def find_cabin_and_class(segment_id, traveler_pricings):
    """
    Match 'segmentId' with travelerPricings->fareDetailsBySegment to find cabin & booking class.
    """
    for tp in traveler_pricings:
        fare_details = tp.get("fareDetailsBySegment", [])
        for fd in fare_details:
            if str(fd.get("segmentId")) == str(segment_id):
                return fd.get("cabin", "N/A"), fd.get("class", "N/A")
    return ("N/A", "N/A")

def parse_itinerary(itinerary_data, traveler_pricings):
    """
    Parse a single itinerary (which can have multiple segments),
    extracting total duration, each segment's departure/arrival, layover, cabin, etc.
    Returns a dict with 'total_duration' and a 'segments' list.
    """
    itinerary_duration_str = itinerary_data.get("duration", "PT0M")
    itinerary_duration = parse_ISO8601_duration(itinerary_duration_str)

    segments = itinerary_data.get("segments", [])
    segment_list = []

    for i, seg in enumerate(segments):
        seg_id = seg.get("id", "")
        carrier_code = seg.get("carrierCode", "")
        flight_number = seg.get("number", "")
        dep_info = seg.get("departure", {})
        arr_info = seg.get("arrival", {})

        dep_airport = dep_info.get("iataCode", "")
        arr_airport = arr_info.get("iataCode", "")
        dep_time_str = dep_info.get("at", "")
        arr_time_str = arr_info.get("at", "")

        segment_duration_str = seg.get("duration", "PT0M")
        segment_duration = parse_ISO8601_duration(segment_duration_str)
        stops = seg.get("numberOfStops", 0)

        # Determine cabin/class from travelerPricings
        cabin, booking_class = find_cabin_and_class(seg_id, traveler_pricings)

        # Calculate layover duration (time between previous arrival and this departure)
        if i == 0:
            layover_duration = "N/A"
        else:
            prev_arr_time = segments[i - 1].get("arrival", {}).get("at", "")
            if prev_arr_time and dep_time_str:
                layover_delta = iso_to_datetime(dep_time_str) - iso_to_datetime(prev_arr_time)
                layover_duration = str(layover_delta)
            else:
                layover_duration = "N/A"

        segment_data = {
            "segment_id": seg_id,
            "carrier_code": carrier_code,
            "flight_number": flight_number,
            "departure_airport": dep_airport,
            "departure_time": dep_time_str,
            "arrival_airport": arr_airport,
            "arrival_time": arr_time_str,
            "segment_duration": str(segment_duration),
            "layover_duration": layover_duration,
            "stops": stops,
            "cabin": cabin,
            "booking_class": booking_class
        }
        segment_list.append(segment_data)

    return {
        "total_duration": str(itinerary_duration),
        "segments": segment_list
    }

def parse_offers_detailed(flight_offers):
    """
    Parse the raw flight offers, returning a list of dictionaries.
    Each entry describes a full round-trip offer:
      - total_price (grandTotal)
      - currency
      - outbound (duration & segments)
      - inbound  (duration & segments)
    """
    all_parsed = []

    for offer in flight_offers:
        price_info = offer.get("price", {})
        grand_total = price_info.get("grandTotal", "N/A")
        currency = price_info.get("currency", "N/A")
        traveler_pricings = offer.get("travelerPricings", [])

        itineraries = offer.get("itineraries", [])
        outbound_data = itineraries[0] if len(itineraries) > 0 else {}
        inbound_data = itineraries[1] if len(itineraries) > 1 else {}

        # Parse outbound & inbound
        parsed_outbound = parse_itinerary(outbound_data, traveler_pricings)
        parsed_inbound = parse_itinerary(inbound_data, traveler_pricings)

        parsed_offer = {
            "total_price": grand_total,
            "currency": currency,
            "outbound": parsed_outbound,
            "inbound": parsed_inbound
        }
        all_parsed.append(parsed_offer)

    return all_parsed


# -----------------------------
# 6. Main Flow
# -----------------------------
if __name__ == "__main__":
    # Example search: round trip from LHR to BOM
    origin = "LHR"
    destination = "BOM"
    departure_date = "2025-06-01"
    return_date = "2025-06-10"
    adults = 1

    logging.info("Fetching flight offers...")

    # Fetch raw offers
    raw_offers = get_flight_offers(
        amadeus_client,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        adults=adults,
        max_results=10000
    )

    if not raw_offers:
        logging.info("No flight offers returned.")
        exit()

    # Parse offers to get detailed itinerary/cost info
    detailed_data = parse_offers_detailed(raw_offers)

    # Write the result to a JSON file
    output_filename = "flight_offers.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2)

    logging.info(f"Flight offers saved to {output_filename}.")
