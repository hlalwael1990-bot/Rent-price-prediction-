import unittest

from werkzeug.datastructures import MultiDict

from src.app_flask import (
    CITY_NEIGHBOURHOOD_MAP,
    build_learning_memory_summary,
    build_prompt_history_digest,
    build_chatbot_system_prompt,
    build_human_explanation,
    build_feature_row_from_inputs,
    build_prediction_inputs,
    compute_min_amenity_increment_local,
    compute_monotonic_price,
    get_effective_accommodates_bounds,
    get_effective_bedrooms_bounds,
    get_other_property_type_price_multiplier,
    compute_property_type_amenity_bonus_local,
    compute_property_type_amenity_factor,
    compute_property_type_floor_price,
    compute_property_type_multiplier,
    compute_fixed_amenity_bonus_local,
    validate_form,
)


class ExplanationTests(unittest.TestCase):
    def test_fixed_bonus_explanation_lists_actual_amenities(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("room_type", "Entire place"),
                ("property_type", "Entire apartment"),
                ("instant_bookable", "No"),
                ("review_scores_rating", "96"),
                ("accommodates", "2"),
                ("amenities", "Dedicated workspace"),
                ("amenities", "TV"),
            ]
        )

        lines = build_human_explanation(
            form=form,
            base_price_local=100.0,
            final_price_local=120.0,
            final_price_output=130.0,
            amenities_count=2,
        )

        self.assertIn(
            "Fixed amenity premium applied for: Dedicated workspace, TV.",
            lines,
        )
        self.assertNotIn(
            "Kitchen applied an additional fixed premium in the final price calculation.",
            lines,
        )

    def test_explanation_does_not_show_raw_model_estimate(self):
        form = MultiDict(
            [
                ("city", "New York"),
                ("neighbourhood", "Allerton"),
                ("room_type", "Entire place"),
                ("property_type", "Entire villa"),
                ("instant_bookable", "No"),
                ("review_scores_rating", "0"),
                ("accommodates", "1"),
            ]
        )

        lines = build_human_explanation(
            form=form,
            base_price_local=54.37,
            final_price_local=230.30,
            final_price_output=230.30,
            amenities_count=0,
        )

        self.assertNotIn("54.37", " ".join(lines))
        self.assertIn(
            "Calibrated estimate in local currency: 230.30 $.",
            lines,
        )

    def test_chatbot_prompt_hides_raw_model_anchor_details(self):
        prompt = build_chatbot_system_prompt(
            listing_snapshot={
                "property_type": "Entire villa",
                "room_type": "Entire place",
            },
            response_language="English",
        )

        self.assertIn("Do not mention internal raw model anchors", prompt)
        self.assertIn("For entire houses and villas", prompt)
        self.assertIn("Never use the `$` symbol for local currency unless the local currency is actually USD.", prompt)
        self.assertIn("show each one with its currency symbol and currency code explicitly", prompt)

    def test_history_digest_includes_listing_and_prediction_context(self):
        digest = build_prompt_history_digest(
            [
                {
                    "role": "user",
                    "content": "Compare this Paris apartment with the previous villa.",
                    "listing_snapshot": {
                        "city": "Paris",
                        "property_type": "Entire apartment",
                        "room_type": "Entire place",
                    },
                    "prediction_snapshot": {
                        "city": "Paris",
                        "formatted_final_price_output": "EUR 210.00 (EUR)",
                    },
                }
            ]
        )

        self.assertIn("Compare this Paris apartment with the previous villa.", digest)
        self.assertIn("listing[city=Paris; property_type=Entire apartment; room_type=Entire place]", digest)
        self.assertIn("prediction[city=Paris; price=EUR 210.00 (EUR)]", digest)

    def test_learning_summary_lists_preferences_and_comparisons(self):
        summary = build_learning_memory_summary(
            {
                "cities": ["Paris", "New York"],
                "comparison_requests": ["Compare Paris and New York for me"],
                "user_goals": ["I want the cheapest option with Wifi"],
            }
        )

        self.assertIn("Cities discussed: Paris; New York", summary)
        self.assertIn("Comparison requests: Compare Paris and New York for me", summary)
        self.assertIn("User goals or preferences: I want the cheapest option with Wifi", summary)

    def test_chatbot_prompt_includes_history_learning_and_comparison_guidance(self):
        prompt = build_chatbot_system_prompt(
            listing_snapshot={"city": "Paris", "property_type": "Entire apartment"},
            response_language="English",
            history=[
                {
                    "role": "user",
                    "content": "Compare this with my last message.",
                    "listing_snapshot": {"city": "Paris", "property_type": "Entire apartment"},
                    "prediction_snapshot": {"city": "Paris", "formatted_final_price_output": "EUR 210.00 (EUR)"},
                }
            ],
            learning_memory={
                "cities": ["Paris"],
                "comparison_requests": ["Compare this with my last message."],
            },
        )

        self.assertIn("Read the stored conversation history carefully before answering.", prompt)
        self.assertIn("Learned user memory from earlier turns:", prompt)
        self.assertIn("Recent stored chat transcript and saved comparison context:", prompt)

    def test_fixed_bonus_is_positive_for_bonus_amenities(self):
        bonus = compute_fixed_amenity_bonus_local("Paris", ["Dedicated workspace", "TV"])
        self.assertGreater(bonus, 0)

    def test_apartment_four_bedrooms_requires_higher_minimum_accommodates(self):
        minimum, maximum = get_effective_accommodates_bounds("Entire apartment", 4)
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 8)

    def test_villa_seven_bedrooms_supports_up_to_twelve_accommodates(self):
        minimum, maximum = get_effective_accommodates_bounds("Entire villa", 7)
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 12)

    def test_house_and_villa_can_start_from_one_accommodate(self):
        house_min, house_max = get_effective_accommodates_bounds("Entire house", 2)
        villa_min, villa_max = get_effective_accommodates_bounds("Entire villa", 3)
        self.assertEqual((house_min, house_max), (1, 6))
        self.assertEqual((villa_min, villa_max), (1, 9))

    def test_villa_requires_at_least_three_bedrooms(self):
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Entire villa", 2)
        self.assertEqual(min_bedrooms, 3)
        self.assertEqual(max_bedrooms, 7)

    def test_condominium_uses_apartment_style_limits(self):
        minimum, maximum = get_effective_accommodates_bounds("Entire condominium", 4)
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 8)

    def test_room_in_hotel_is_single_bedroom_and_small_capacity(self):
        minimum, maximum = get_effective_accommodates_bounds("Room in hotel", 1)
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 3)
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Room in hotel", 2)
        self.assertEqual(min_bedrooms, 1)
        self.assertEqual(max_bedrooms, 1)

    def test_shared_room_is_single_bedroom_and_small_capacity(self):
        minimum, maximum = get_effective_accommodates_bounds("Other", 1, "Guest house", "Shared room")
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 3)
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Other", 2, "Guest house", "Shared room")
        self.assertEqual(min_bedrooms, 1)
        self.assertEqual(max_bedrooms, 1)

    def test_other_type_uses_moderate_limits(self):
        minimum, maximum = get_effective_accommodates_bounds("Other", 5)
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 8)

    def test_studio_uses_single_bedroom_limits(self):
        minimum, maximum = get_effective_accommodates_bounds("Other", 1, "Studio")
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 3)
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Other", 2, "Studio")
        self.assertEqual(min_bedrooms, 1)
        self.assertEqual(max_bedrooms, 1)

    def test_cabin_uses_requested_limits(self):
        minimum, maximum = get_effective_accommodates_bounds("Other", 3, "Cabin")
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 6)
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Other", 4, "Cabin")
        self.assertEqual(min_bedrooms, 2)
        self.assertEqual(max_bedrooms, 3)

    def test_chalet_uses_requested_limits(self):
        minimum, maximum = get_effective_accommodates_bounds("Other", 3, "Chalet")
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 6)
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Other", 4, "Chalet")
        self.assertEqual(min_bedrooms, 2)
        self.assertEqual(max_bedrooms, 3)

    def test_loft_uses_zero_bedroom_limits(self):
        minimum, maximum = get_effective_accommodates_bounds("Other", 0, "Loft")
        self.assertEqual(minimum, 1)
        self.assertEqual(maximum, 3)
        min_bedrooms, max_bedrooms = get_effective_bedrooms_bounds("Other", 2, "Loft")
        self.assertEqual(min_bedrooms, 0)
        self.assertEqual(max_bedrooms, 0)

    def test_validation_rejects_invalid_room_type_for_entire_condominium(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Entire condominium"),
                ("room_type", "Hotel room"),
                ("instant_bookable", "No"),
                ("accommodates", "4"),
                ("bedrooms", "4"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Entire condominium, room type must be one of: ['Entire place'].")

    def test_validation_accepts_room_in_hotel_with_hotel_room(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Room in hotel"),
                ("room_type", "Hotel room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertIsNone(message)

    def test_validation_rejects_private_room_for_room_in_hotel(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Room in hotel"),
                ("room_type", "Private room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Room in hotel, room type must be one of: ['Hotel room'].")

    def test_validation_accepts_private_room_for_entire_apartment(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Entire apartment"),
                ("room_type", "Private room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertIsNone(message)

    def test_validation_accepts_private_room_for_entire_house(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Entire house"),
                ("room_type", "Private room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "2"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertIsNone(message)

    def test_validation_requires_specific_other_property_type(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Private room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", ""),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Other, please choose one of: ['Studio', 'Chalet', 'Guest house', 'Cabin', 'Loft'].")

    def test_validation_accepts_specific_other_property_type(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Studio"),
            ]
        )

        message = validate_form(form)
        self.assertIsNone(message)

    def test_validation_rejects_non_entire_place_for_studio(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Private room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Studio"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Other, room type must be one of: ['Entire place'].")

    def test_validation_rejects_studio_with_more_than_one_bedroom(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "2"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Studio"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Studio, bedrooms must be 1.")

    def test_validation_rejects_cabin_with_too_many_bedrooms(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "4"),
                ("bedrooms", "4"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Cabin"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Cabin, bedrooms must be between 2 and 3.")

    def test_validation_rejects_cabin_with_too_many_accommodates(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "7"),
                ("bedrooms", "3"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Cabin"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Cabin, accommodates must be between 1 and 6.")

    def test_validation_rejects_chalet_with_too_many_bedrooms(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "4"),
                ("bedrooms", "4"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Chalet"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Chalet, bedrooms must be between 2 and 3.")

    def test_validation_rejects_loft_with_non_zero_bedrooms(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Loft"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Loft, bedrooms must be 0.")

    def test_validation_rejects_hotel_room_for_guest_house(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Hotel room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Guest house"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(
            message,
            "For Other, room type must be one of: ['Entire place', 'Private room', 'Shared room'].",
        )

    def test_validation_rejects_hotel_room_for_cabin(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Hotel room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Cabin"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(
            message,
            "For Other, room type must be one of: ['Entire place', 'Private room', 'Shared room'].",
        )

    def test_validation_rejects_shared_room_with_more_than_one_bedroom(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("room_type", "Shared room"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "2"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
                ("other_property_type", "Guest house"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Shared room, bedrooms must be 1.")

    def test_build_prediction_inputs_keeps_other_property_type(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("other_property_type", "Studio"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "1"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        inputs = build_prediction_inputs(form)
        self.assertEqual(inputs["other_property_type"], "Studio")

    def test_other_property_type_multipliers_follow_expected_order(self):
        self.assertLess(
            get_other_property_type_price_multiplier("Studio"),
            get_other_property_type_price_multiplier("Guest house"),
        )
        self.assertLess(
            get_other_property_type_price_multiplier("Guest house"),
            get_other_property_type_price_multiplier("Loft"),
        )
        self.assertLess(
            get_other_property_type_price_multiplier("Loft"),
            get_other_property_type_price_multiplier("Cabin"),
        )
        self.assertLess(
            get_other_property_type_price_multiplier("Cabin"),
            get_other_property_type_price_multiplier("Chalet"),
        )

    def test_other_property_type_changes_price(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Other",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 1,
            "bedrooms": 1,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        _, studio_price = compute_monotonic_price({**base_inputs, "other_property_type": "Studio"})
        _, chalet_price = compute_monotonic_price({**base_inputs, "other_property_type": "Chalet"})

        self.assertGreater(chalet_price, studio_price)

    def test_studio_entire_place_is_less_expensive_than_entire_apartment(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 2,
            "bedrooms": 1,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        _, apartment_price = compute_monotonic_price({**base_inputs, "property_type": "Entire apartment"})
        _, studio_price = compute_monotonic_price(
            {**base_inputs, "property_type": "Other", "other_property_type": "Studio"}
        )

        self.assertLess(studio_price, apartment_price)
        self.assertGreaterEqual(apartment_price, round(studio_price * 1.30, 2))

    def test_entire_apartment_is_more_expensive_than_shared_room_for_similar_inputs(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "instant_bookable": "No",
            "accommodates": 2,
            "bedrooms": 1,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        _, apartment_price = compute_monotonic_price(
            {**base_inputs, "property_type": "Entire apartment", "room_type": "Entire place"}
        )
        _, shared_room_price = compute_monotonic_price(
            {
                **base_inputs,
                "property_type": "Other",
                "other_property_type": "Guest house",
                "room_type": "Shared room",
            }
        )

        self.assertGreater(apartment_price, shared_room_price)

    def test_shared_room_is_at_most_half_of_entire_place_for_similar_inputs(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Other",
            "other_property_type": "Guest house",
            "instant_bookable": "No",
            "accommodates": 2,
            "bedrooms": 1,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        _, entire_place_price = compute_monotonic_price({**base_inputs, "room_type": "Entire place"})
        _, shared_room_price = compute_monotonic_price({**base_inputs, "room_type": "Shared room"})

        self.assertLessEqual(shared_room_price, entire_place_price * 0.5)

    def test_explanation_mentions_other_property_type_adjustment(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Other"),
                ("other_property_type", "Chalet"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "2"),
                ("bedrooms", "1"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        lines = build_human_explanation(
            form=form,
            base_price_local=100.0,
            final_price_local=118.0,
            final_price_output=118.0,
            amenities_count=0,
        )

        joined = " ".join(lines)
        self.assertIn("Specific other property type selected: Chalet", joined)
        self.assertIn("premium adjustment", joined)

    def test_validation_accepts_apartment_four_bedrooms_with_one_guest(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Entire apartment"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "1"),
                ("bedrooms", "4"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertIsNone(message)

    def test_validation_rejects_villa_with_two_bedrooms(self):
        form = MultiDict(
            [
                ("city", "Paris"),
                ("neighbourhood", "Batignolles-Monceau"),
                ("property_type", "Entire villa"),
                ("room_type", "Entire place"),
                ("instant_bookable", "No"),
                ("accommodates", "4"),
                ("bedrooms", "2"),
                ("minimum_nights", "2"),
                ("maximum_nights", "30"),
                ("review_scores_rating", "96"),
            ]
        )

        message = validate_form(form)
        self.assertEqual(message, "For Entire villa, bedrooms must be between 3 and 7.")

    def test_villa_price_does_not_drop_when_accommodates_increase(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Entire villa",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 9,
            "bedrooms": 7,
            "minimum_nights": 5,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": ["Wifi"],
        }

        _, price_9 = compute_monotonic_price(base_inputs)
        _, price_10 = compute_monotonic_price({**base_inputs, "accommodates": 10})
        _, price_11 = compute_monotonic_price({**base_inputs, "accommodates": 11})
        _, price_12 = compute_monotonic_price({**base_inputs, "accommodates": 12})

        self.assertGreaterEqual(price_10, price_9)
        self.assertGreaterEqual(price_11, price_10)
        self.assertGreaterEqual(price_12, price_11)

    def test_metadata_uses_clean_mexico_city_neighbourhood_name(self):
        neighbourhoods = CITY_NEIGHBOURHOOD_MAP["Mexico City"]
        self.assertTrue(
            any("lvaro Obregon" in item for item in neighbourhoods),
            neighbourhoods,
        )


    def test_villa_engineered_features_are_populated(self):
        row = build_feature_row_from_inputs(
            {
                "city": "Paris",
                "neighbourhood": "Batignolles-Monceau",
                "property_type": "Entire villa",
                "room_type": "Entire place",
                "instant_bookable": "No",
                "accommodates": 8,
                "bedrooms": 4,
                "minimum_nights": 2,
                "maximum_nights": 30,
                "review_scores_rating": 96,
                "amenities": ["Wifi", "Kitchen", "TV"],
            }
        ).iloc[0]

        self.assertEqual(int(row["is_villa"]), 1)
        self.assertEqual(int(row["is_house"]), 0)
        self.assertEqual(float(row["villa_x_accommodates"]), 8.0)
        self.assertEqual(float(row["villa_x_bedrooms"]), 4.0)
        self.assertEqual(float(row["villa_x_rating"]), 96.0)

    def test_property_type_multiplier_prioritizes_house_and_villa(self):
        apartment_multiplier = compute_property_type_multiplier(
            property_type="Entire apartment",
            room_type="Entire place",
            accommodates=6,
            bedrooms=3,
        )
        house_multiplier = compute_property_type_multiplier(
            property_type="Entire house",
            room_type="Entire place",
            accommodates=6,
            bedrooms=3,
        )
        villa_multiplier = compute_property_type_multiplier(
            property_type="Entire villa",
            room_type="Entire place",
            accommodates=6,
            bedrooms=3,
        )

        self.assertGreater(house_multiplier, apartment_multiplier)
        self.assertGreater(villa_multiplier, house_multiplier)

    def test_property_type_floor_price_is_higher_for_villa_than_house(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Entire villa",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 8,
            "bedrooms": 4,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": ["Wifi", "Kitchen", "TV"],
        }

        house_floor = compute_property_type_floor_price(
            base_inputs={**base_inputs, "property_type": "Entire house"},
            selected_amenities=base_inputs["amenities"],
            review_score=96,
            city="Paris",
            property_type="Entire house",
            room_type="Entire place",
            accommodates=8,
            bedrooms=4,
        )
        villa_floor = compute_property_type_floor_price(
            base_inputs=base_inputs,
            selected_amenities=base_inputs["amenities"],
            review_score=96,
            city="Paris",
            property_type="Entire villa",
            room_type="Entire place",
            accommodates=8,
            bedrooms=4,
        )

        self.assertGreater(house_floor, 0.0)
        self.assertGreater(villa_floor, house_floor)

    def test_all_property_types_use_same_amenity_factor(self):
        apartment_factor = compute_property_type_amenity_factor("Entire apartment", "Entire place")
        house_factor = compute_property_type_amenity_factor("Entire house", "Entire place")
        villa_factor = compute_property_type_amenity_factor("Entire villa", "Entire place")

        self.assertEqual(house_factor, apartment_factor)
        self.assertEqual(villa_factor, apartment_factor)

    def test_villa_floor_price_does_not_decrease_when_wifi_is_added(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Entire villa",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 8,
            "bedrooms": 4,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        no_wifi_price = compute_property_type_floor_price(
            base_inputs=base_inputs,
            selected_amenities=[],
            review_score=96,
            city="Paris",
            property_type="Entire villa",
            room_type="Entire place",
            accommodates=8,
            bedrooms=4,
        )
        wifi_price = compute_property_type_floor_price(
            base_inputs={**base_inputs, "amenities": ["Wifi"]},
            selected_amenities=["Wifi"],
            review_score=96,
            city="Paris",
            property_type="Entire villa",
            room_type="Entire place",
            accommodates=8,
            bedrooms=4,
        )

        self.assertGreaterEqual(wifi_price, no_wifi_price)

    def test_monotonic_price_does_not_drop_when_wifi_and_kitchen_are_added(self):
        wifi_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Entire villa",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 8,
            "bedrooms": 4,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": ["Wifi"],
        }
        wifi_kitchen_inputs = {**wifi_inputs, "amenities": ["Wifi", "Kitchen"]}

        _, wifi_price = compute_monotonic_price(wifi_inputs)
        _, wifi_kitchen_price = compute_monotonic_price(wifi_kitchen_inputs)

        self.assertGreater(wifi_kitchen_price, wifi_price)

    def test_house_and_villa_use_positive_minimum_amenity_increment(self):
        house_increment = compute_min_amenity_increment_local("Paris", "Entire house", "Wifi")
        villa_increment = compute_min_amenity_increment_local("Paris", "Entire villa", "Kitchen")
        tv_increment = compute_min_amenity_increment_local("Paris", "Entire villa", "TV")

        self.assertGreater(house_increment, 0.0)
        self.assertGreater(villa_increment, house_increment)
        self.assertLess(tv_increment, 10.0)

    def test_entire_house_hot_water_increases_price(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Entire house",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 6,
            "bedrooms": 3,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        _, base_price = compute_monotonic_price(base_inputs)
        _, hot_water_price = compute_monotonic_price({**base_inputs, "amenities": ["Hot water"]})

        self.assertGreater(hot_water_price, base_price)

    def test_entire_house_tv_adds_small_logical_increment(self):
        base_inputs = {
            "city": "Paris",
            "neighbourhood": "Batignolles-Monceau",
            "property_type": "Entire house",
            "room_type": "Entire place",
            "instant_bookable": "No",
            "accommodates": 6,
            "bedrooms": 3,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "review_scores_rating": 96,
            "amenities": [],
        }

        _, base_price = compute_monotonic_price(base_inputs)
        _, tv_price = compute_monotonic_price({**base_inputs, "amenities": ["TV"]})
        logical_tv_bonus = compute_property_type_amenity_bonus_local("Paris", "Entire house", ["TV"])

        self.assertGreater(tv_price, base_price)
        self.assertLessEqual(round(tv_price - base_price, 2), round(logical_tv_bonus, 2))


if __name__ == "__main__":
    unittest.main()
