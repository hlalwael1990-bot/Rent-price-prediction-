import unittest

from werkzeug.datastructures import MultiDict

from src.app_flask import (
    CITY_NEIGHBOURHOOD_MAP,
    build_chatbot_system_prompt,
    build_human_explanation,
    build_feature_row_from_inputs,
    compute_min_amenity_increment_local,
    compute_monotonic_price,
    compute_property_type_amenity_bonus_local,
    compute_property_type_amenity_factor,
    compute_property_type_floor_price,
    compute_property_type_multiplier,
    compute_fixed_amenity_bonus_local,
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
            "Calibrated estimate in local currency: 230.30.",
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

    def test_fixed_bonus_is_positive_for_bonus_amenities(self):
        bonus = compute_fixed_amenity_bonus_local("Paris", ["Dedicated workspace", "TV"])
        self.assertGreater(bonus, 0)

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
