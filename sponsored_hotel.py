# Example: Tokyo (dest_id -246227, matching the sample search URL you shared:
# .../searchresults.html?dest_id=-246227&dest_type=city...). Replace/extend
# this with your actual destinations and sponsored partners.
#
# NOTE on field names, which hotels._parse_hotels/search_mock_hotels read:
# * "hotel_name" is the key the StayAPI search shape uses ("name" also works).
# * "price_per_night" — NOT "price" — because these figures are nightly rates.
#   StayAPI's own "price" field is the total for the whole stay, so putting a
#   nightly rate there made _parse_lead_price divide it by the night count.
SPONSORED_MOCK_HOTELS: list[dict] = [
    {
        "dest_id": "-246227",
        "hotel_id": "sponsored-tokyo-001",
        "hotel_name": "The Ginza Grand Hotel & Spa",
        "address": "5-4-1 Ginza",
        "city": "Chuo City",
        "latitude": 35.6717,
        "longitude": 139.7650,
        "star_rating": 5,
        "price_per_night": 245.00,
        "currency": "USD",
        "rating": 9.1,
        "image_url": "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=800",
        "free_cancellation": True,
        "no_prepayment": True,
        "is_sold_out": False,
        "room_name": "Sponsored Rate",
    },
    {
        "dest_id": "-246227",
        "hotel_id": "sponsored-tokyo-002",
        "hotel_name": "Shinjuku Skyline Suites",
        "address": "2-1-1 Nishi-Shinjuku",
        "city": "Shinjuku City",
        "latitude": 35.6896,
        "longitude": 139.6917,
        "star_rating": 4,
        "price_per_night": 168.50,
        "currency": "USD",
        "rating": 8.7,
        "image_url": "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?w=800",
        "free_cancellation": True,
        "no_prepayment": False,
        "is_sold_out": False,
        "room_name": "Sponsored Rate",
    },
    {
        "dest_id": "-246227",
        "hotel_id": "sponsored-tokyo-003",
        "hotel_name": "Asakusa Riverside Ryokan",
        "address": "1-3-2 Asakusa",
        "city": "Taito City",
        "latitude": 35.7118,
        "longitude": 139.7966,
        "star_rating": 4,
        "price_per_night": 132.00,
        "currency": "USD",
        "rating": 9.3,
        "image_url": "https://images.unsplash.com/photo-1590490360182-c33d57733427?w=800",
        "free_cancellation": True,
        "no_prepayment": True,
        "is_sold_out": False,
        "room_name": "Sponsored Rate",
    },
]

# ==========================================
# Non-sponsored (regular) mock hotels
# ==========================================
# Same StayAPI-shaped fields as SPONSORED_MOCK_HOTELS above (see the field-name
# notes at the top of this file). The only functional difference is
# "is_sponsored": False, which hotels.search_mock_hotels reads per-item to set
# Hotel.is_sponsored — the frontend (ChangeAccommodationModal) splits results
# into "featuredStays" vs "regularStays" based on that flag.
NOT_SPONSORED_MOCK_HOTELS: list[dict] = [
    {
        "dest_id": "-246227",
        "hotel_id": "regular-tokyo-001",
        "hotel_name": "Akasaka Business Hotel",
        "address": "3-8-5 Akasaka",
        "city": "Minato City",
        "latitude": 35.6735,
        "longitude": 139.7368,
        "star_rating": 3,
        "price_per_night": 89.00,
        "currency": "USD",
        "rating": 8.2,
        "image_url": "https://images.unsplash.com/photo-1590073844006-33379778ae09?w=800",
        "free_cancellation": True,
        "no_prepayment": True,
        "is_sold_out": False,
        "room_name": "Standard Double Room",
        "is_sponsored": False,
    },
    {
        "dest_id": "-246227",
        "hotel_id": "regular-tokyo-002",
        "hotel_name": "Ueno Park Hotel",
        "address": "6-1-2 Ueno",
        "city": "Taito City",
        "latitude": 35.7141,
        "longitude": 139.7774,
        "star_rating": 3,
        "price_per_night": 76.50,
        "currency": "USD",
        "rating": 7.9,
        "image_url": "https://images.unsplash.com/photo-1611892440504-42a792e24d32?w=800",
        "free_cancellation": False,
        "no_prepayment": True,
        "is_sold_out": False,
        "room_name": "Twin Room",
        "is_sponsored": False,
    },
    {
        "dest_id": "-246227",
        "hotel_id": "regular-tokyo-003",
        "hotel_name": "Ikebukuro Comfort Inn",
        "address": "1-14-3 Ikebukuro",
        "city": "Toshima City",
        "latitude": 35.7295,
        "longitude": 139.7109,
        "star_rating": 3,
        "price_per_night": 102.00,
        "currency": "USD",
        "rating": 8.4,
        "image_url": "https://images.unsplash.com/photo-1584132967334-10e028bd69f7?w=800",
        "free_cancellation": True,
        "no_prepayment": False,
        "is_sold_out": False,
        "room_name": "Deluxe Room",
        "is_sponsored": False,
    },
    {
        "dest_id": "-246227",
        "hotel_id": "regular-tokyo-004",
        "hotel_name": "Shibuya City Lodge",
        "address": "2-22-8 Shibuya",
        "city": "Shibuya City",
        "latitude": 35.6591,
        "longitude": 139.7005,
        "star_rating": 4,
        "price_per_night": 145.00,
        "currency": "USD",
        "rating": 8.8,
        "image_url": "https://images.unsplash.com/photo-1611892440504-42a792e24d32?w=800",
        "free_cancellation": True,
        "no_prepayment": True,
        "is_sold_out": False,
        "room_name": "Superior Room",
        "is_sponsored": False,
    },
    {
        "dest_id": "-246227",
        "hotel_id": "regular-tokyo-005",
        "hotel_name": "Ryogoku Sumo Inn",
        "address": "4-2-1 Ryogoku",
        "city": "Sumida City",
        "latitude": 35.6969,
        "longitude": 139.7933,
        "star_rating": 2,
        "price_per_night": 58.00,
        "currency": "USD",
        "rating": 7.5,
        "image_url": "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=800",
        "free_cancellation": False,
        "no_prepayment": False,
        "is_sold_out": True,
        "room_name": "Economy Room",
        "is_sponsored": False,
    },
    {
        "dest_id": "-246227",
        "hotel_id": "regular-tokyo-006",
        "hotel_name": "Odaiba Bay View Hotel",
        "address": "1-6-1 Daiba",
        "city": "Minato City",
        "latitude": 35.6267,
        "longitude": 139.7746,
        "star_rating": 4,
        "price_per_night": 178.00,
        "currency": "USD",
        "rating": 8.9,
        "image_url": "https://images.unsplash.com/photo-1590490360182-c33d57733427?w=800",
        "free_cancellation": True,
        "no_prepayment": True,
        "is_sold_out": False,
        "room_name": "Bay View Room",
        "is_sponsored": False,
    },
]

# ==========================================
# Mock per-hotel room prices
# ==========================================
# Raw dicts shaped like StayAPI's /v1/booking/hotel/prices response (a
# "rooms" list of blocks), NOT RoomOption objects — so this can be run
# through hotels._parse_room_options(), the same parser real StayAPI room
# data goes through. No second parsing path, matching how search_mock_hotels
# reuses the field-reading helpers rather than duplicating them.
#
# Three tiers are generated per hotel from its own price_per_night (base /
# +30% / +70%), so /hotel/{hotel_id}/prices has more than a single room to
# offer, same as a real StayAPI response would.
def _mock_room_blocks(hotel: dict) -> dict:
    base = hotel["price_per_night"]
    currency = hotel.get("currency", "USD")
    free_cancel = bool(hotel.get("free_cancellation", False))
    return {
        "rooms": [
            {
                "room_name": hotel.get("room_name") or "Standard Room",
                "price_per_night": round(base, 2),
                "currency": currency,
                "max_occupancy": 2,
                "breakfast_included": False,
                "is_refundable": free_cancel,
                "cancellation_policy": (
                    "Free cancellation until 24h before check-in" if free_cancel else "Non-refundable"
                ),
            },
            {
                "room_name": "Deluxe Room",
                "price_per_night": round(base * 1.3, 2),
                "currency": currency,
                "max_occupancy": 3,
                "breakfast_included": True,
                "is_refundable": True,
                "cancellation_policy": "Free cancellation until 24h before check-in",
            },
            {
                "room_name": "Executive Suite",
                "price_per_night": round(base * 1.7, 2),
                "currency": currency,
                "max_occupancy": 4,
                "breakfast_included": True,
                "is_refundable": False,
                "cancellation_policy": "Non-refundable",
            },
        ]
    }


# Keyed by hotel_id so api.py can look prices up the same way
# get_hotel_prices_raw(hotel_id, ...) would key a real StayAPI call.
# Built from both lists, so adding a hotel above automatically gets
# room-price coverage here too.
MOCK_ROOM_PRICES_RAW: dict[str, dict] = {
    h["hotel_id"]: _mock_room_blocks(h)
    for h in SPONSORED_MOCK_HOTELS + NOT_SPONSORED_MOCK_HOTELS
}