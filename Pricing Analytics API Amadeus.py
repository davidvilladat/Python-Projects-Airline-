import logging
import json
from datetime import datetime, timedelta
import re

from amadeus import Client, ResponseError

# ------------------------------------------------------------
# 1. Configure Logging
# ------------------------------------------------------------
# Sets up basic logging for the script, so that messages can 
# be displayed in the console. The 'format' specifies how 
# each log message should look.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# 2. Amadeus Credentials
# ------------------------------------------------------------
# Provide your Amadeus API credentials here. 
# In a production environment, avoid hardcoding sensitive info.
amadeus_client = Client(
    client_id='pNYUh6kEI8K4JbqkXl79xCZcI2IMbs37',
    client_secret='v8hJNTTa1DfLvXG7'
)

# ------------------------------------------------------------
# 3. Initialize Amadeus Client
# ------------------------------------------------------------
# (Already initialized above in 'amadeus_client')

# ------------------------------------------------------------
# 4. Fetch Flight Offers
# ------------------------------------------------------------
def get_flight_offers(amadeus_client, origin, destination,
                      departure_date, return_date,
                      adults=1, max_results=10):
    """
    Retrieves flight offers from Amadeus for round-trip flights.
    
    :param amadeus_client: The authenticated Amadeus client.
    :param origin: Origin airport code (e.g., 'LHR').
    :param destination: Destination airport code (e.g., 'BOM').
    :param departure_date: Departure date in 'YYYY-MM-DD' format.
    :param return_date: Return date in 'YYYY-MM-DD' format.
    :param adults: Number of adult passengers.
    :param max_results: Maximum number of flight offers to retrieve.
    :return: A list of raw flight offers in JSON format.
    """
    try:
        # Calls the Amadeus API to fetch flight offers based on 
        # specified query parameters.
        response = amadeus_client.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            returnDate=return_date,
            adults=adults,
            max=max_results
        )
        return response.data  # The offers are returned as JSON data.
    except ResponseError as error:
        # If an error occurs while calling the API, log the error message.
        logging.error(f"Error while fetching flight offers: {error}")
        return []

# ------------------------------------------------------------
# 5. Helper Parsing Functions
# ------------------------------------------------------------
def iso_to_datetime(iso_str):
    """
    Convert an ISO 8601 date/time string (e.g., '2025-06-01T10:15:00') 
    into a Python datetime object.
    """
    return datetime.fromisoformat(iso_str)

def parse_ISO8601_duration(duration_str):
    """
    Parse an ISO 8601 duration string (e.g., 'PT14H20M') into 
    a timedelta object representing hours/minutes/seconds.
    
    For example:
      'PT2H10M' => 2 hours, 10 minutes.
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
    Given a segment ID and traveler pricing data, determine the 
    cabin (e.g., Economy, Business) and booking class (e.g., 'Y', 'B') 
    for that particular flight segment.
    
    :param segment_id: Unique ID for a flight segment.
    :param traveler_pricings: Pricing details at the traveler level.
    :return: A tuple (cabin, booking_class).
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
    extracting the total duration and each segment's details, 
    such as departure/arrival times, layover, cabin, etc.
    
    :param itinerary_data: Dictionary containing itinerary details (duration, segments).
    :param traveler_pricings: List of pricing details for travelers, used for cabin/booking class info.
    :return: A dictionary with 'total_duration' and a 'segments' list.
    """
    # Overall itinerary duration (includes all segments).
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

        # Determine the cabin/class for this segment.
        cabin, booking_class = find_cabin_and_class(seg_id, traveler_pricings)

        # Calculate layover duration compared to the previous segment.
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
    Take raw flight offers and transform them into a structured list 
    that includes total price, currency, and details for outbound/inbound 
    itineraries (segments, duration, cabin, etc.).
    
    :param flight_offers: The raw offers (list of dictionaries) returned from the API.
    :return: A list of parsed flight offers with key information.
    """
    all_parsed = []

    for offer in flight_offers:
        price_info = offer.get("price", {})
        grand_total = price_info.get("grandTotal", "N/A")
        currency = price_info.get("currency", "N/A")
        traveler_pricings = offer.get("travelerPricings", [])

        # Each offer can contain multiple itineraries (e.g., outbound, inbound).
        itineraries = offer.get("itineraries", [])
        outbound_data = itineraries[0] if len(itineraries) > 0 else {}
        inbound_data = itineraries[1] if len(itineraries) > 1 else {}

        # Parse outbound & inbound itineraries separately.
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

# ------------------------------------------------------------
# 6. Main Flow
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example search parameters: Round trip from LHR (London Heathrow) 
    # to BOM (Mumbai), departing on 2025-06-01 and returning on 2025-06-10.
    origin = "LHR"
    destination = "BOM"
    departure_date = "2025-06-01"
    return_date = "2025-06-10"
    adults = 1

    logging.info("Fetching flight offers...")

    # Fetch raw offers from the Amadeus API.
    raw_offers = get_flight_offers(
        amadeus_client,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        adults=adults,
        max_results=10000
    )

    # If there are no offers, log a message and exit.
    if not raw_offers:
        logging.info("No flight offers returned.")
        exit()

    # Parse the offers to get a more detailed data structure.
    detailed_data = parse_offers_detailed(raw_offers)

    # Write the result to a JSON file for inspection or further processing.
    output_filename = "flight_offers.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2)

    logging.info(f"Flight offers saved to {output_filename}.")
