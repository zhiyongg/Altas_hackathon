def get_category(google_type: str) -> str:
    google_type = google_type.lower()

    # Culture
    if google_type in [
        "museum", "history_museum", "art_museum", "historical_place",
        "historical_landmark", "cultural_landmark", "castle", "monument",
        "art_gallery", "place_of_worship"
    ]:
        return "Culture"

    # Scenery
    elif google_type in [
        "national_park", "state_park", "park", "botanical_garden",
        "nature_preserve", "scenic_spot", "beach", "lake", "mountain_peak",
        "river", "wildlife_park", "zoo", "aquarium"
    ]:
        return "Scenery"

    # Food
    elif google_type in [
        "restaurant", "cafe", "bakery", "food_court", "fine_dining_restaurant",
        "family_restaurant"
    ] or "street_food" in google_type or "cuisine" in google_type:
        return "Food"

    # Shopping
    elif google_type in [
        "shopping_mall", "market", "department_store", "gift_shop",
        "book_store", "flea_market", "farmers_market"
    ] or "clothing" in google_type or "fashion" in google_type:
        return "Shopping"

    # Entertainment
    elif google_type in [
        "amusement_park", "water_park", "movie_theater", "bowling_alley",
        "karaoke", "concert_hall", "live_music_venue", "comedy_club", "arcade",
        "night_club", "bar", "cocktail_bar", "pub", "wine_bar", "lounge_bar",
        "indoor_playground", "playground"
    ]:
        return "Entertainment"

    # Adventure
    elif google_type in [
        "hiking_area", "adventure_sports_center", "cycling_park",
        "ski_resort", "golf_course", "sports_activity_location",
        "swimming_pool", "fishing_charter"
    ]:
        return "Adventure"

    # Wellness
    elif google_type in [
        "spa", "massage", "sauna", "wellness_center", "yoga_studio"
    ]:
        return "Wellness"

    # City
    elif google_type in [
        "tourist_attraction", "observation_deck", "plaza", "landmark",
        "point_of_interest"
    ]:
        return "City"

    else:
        return "Uncategorized"


def calculate_preference_scores(preferences, selected_preferences):
    """
    Calculate initial user preference scores.

    Rules:
    - Every preference starts at 0.50
    - Selected preferences receive a higher score
    - Non-selected preferences receive a slight decrease
    - Scores are bounded between 0.0 and 1.0

    Parameters:
        preferences (list): All available preference categories.
        selected_preferences (list): Categories selected by the user.

    Returns:
        dict: Preference scores.
    """

    total_preferences = len(preferences)
    selected_count = len(selected_preferences)

    scores = {}

    for preference in preferences:

        if preference in selected_preferences:
            # Selected preference gets a stronger score
            score = 0.50 + 0.50 * (1 - selected_count / total_preferences)

        else:
            # Non-selected preference gets a slight decrease
            score = 0.50 - 0.20 * (selected_count / total_preferences)

        # Keep score between 0 and 1
        score = max(0.0, min(1.0, score))

        scores[preference] = round(score, 4)

    return scores