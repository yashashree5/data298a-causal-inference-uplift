"""Unit tests for weather alignment and sun position. No network or nuPlan data is used."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
except ImportError:  # the fast suite must still run without pandas
    pd = None

if pd is not None:
    from tools.extract_weather_features import (  # noqa: E402
        FEATURES, HOURLY, PREV_3H, SUN, WEATHER_MODEL, hourly_frame, request_params, scenario_weather,
        sun_elevation_deg,
    )

needs_pandas = unittest.skipUnless(pd, "pandas is required")
H = 3_600_000_000


def us(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1_000_000)


def raw_response(city: str, start: str, hours: int, precipitation: list[float], gap_at: int | None = None) -> dict:
    times = list(pd.date_range(start, periods=hours, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M"))
    if gap_at is not None:
        del times[gap_at]
    n = len(times)
    hourly = {"time": times}
    for var in HOURLY:
        hourly[var] = [0.0] * n
    hourly["precipitation"] = precipitation[:n]
    hourly["rain"] = [p / 2 for p in precipitation[:n]]
    hourly["weather_code"] = [61 if p > 0 else 0 for p in precipitation[:n]]
    hourly["temperature_2m"] = [20.0 + i for i in range(n)]  # identifies the hour that was joined
    return {"cities": {city: {"response": {"hourly": hourly}}}}


@needs_pandas
class SunElevationTest(unittest.TestCase):
    def max_elevation(self, day: tuple, lat: float, lon: float) -> float:
        minutes = np.arange(24 * 60) * 60_000_000 + us(*day)
        return float(sun_elevation_deg(minutes, lat, lon).max())

    def test_equinox_noon_elevation_is_90_minus_latitude(self):
        for lat, lon in ((1.2997, 103.79157), (42.34636, -71.04257)):
            self.assertAlmostEqual(self.max_elevation((2021, 3, 20), lat, lon), 90 - lat, delta=0.7)

    def test_june_solstice_noon_adds_axial_tilt(self):
        lat = 42.34636
        self.assertAlmostEqual(self.max_elevation((2021, 6, 21), lat, -71.04257), 90 - lat + 23.44, delta=0.7)

    def test_local_midnight_is_below_horizon(self):
        # 04:00 UTC is midnight in Boston in summer (UTC-4).
        self.assertLess(float(sun_elevation_deg(us(2021, 8, 20, 4), 42.34636, -71.04257)), -20)


@needs_pandas
class AlignmentTest(unittest.TestCase):
    def setUp(self):
        precipitation = [0, 0, 1.0, 2.0, 0, 0, 0.5, 0]
        self.hourly = hourly_frame(raw_response("boston", "2021-08-20 08:00", 8, precipitation))
        self.points = {"boston": {"latitude": 42.34636, "longitude": -71.04257}}

    def scen(self, starts):
        return pd.DataFrame({"scenario_id": [f"log:{i:016x}" for i in range(len(starts))], "log_name": "log",
                             "scenario_token": [f"{i:016x}" for i in range(len(starts))],
                             "source_city": "boston", "window_start_us": starts})

    def test_three_hour_sum_uses_only_past_labels(self):
        h = self.hourly.set_index(self.hourly["hour_utc"].dt.hour)
        self.assertTrue(pd.isna(h.loc[8, PREV_3H]) and pd.isna(h.loc[9, PREV_3H]))
        self.assertEqual(h.loc[10, PREV_3H], 1.0)   # labels 8, 9, 10
        self.assertEqual(h.loc[11, PREV_3H], 3.0)   # labels 9, 10, 11
        self.assertEqual(h.loc[13, PREV_3H], 2.0)   # labels 11, 12, 13
        self.assertEqual(h.loc[14, PREV_3H], 0.5)

    def test_scenario_uses_hour_at_or_before_start(self):
        starts = [us(2021, 8, 20, 11, 0, 0), us(2021, 8, 20, 11, 59, 59), us(2021, 8, 20, 12, 0, 0)]
        df = scenario_weather(self.scen(starts), self.hourly, self.points)
        # temperature encodes the joined hour: 20 + (hour - 8)
        self.assertEqual(df["weather_temperature_c"].tolist(), [23.0, 23.0, 24.0])
        self.assertEqual(df["weather_precipitation_mm"].tolist(), [2.0, 2.0, 0.0])
        self.assertEqual(df[PREV_3H].tolist(), [3.0, 3.0, 3.0])

    def test_missing_hours_are_null_with_indicator_never_imputed(self):
        df = scenario_weather(self.scen([us(2021, 8, 20, 9, 30), us(2021, 8, 21, 9, 30)]), self.hourly, self.points)
        self.assertTrue(df[f"{PREV_3H}_missing"].tolist() == [True, True])
        self.assertTrue(df["weather_temperature_c_missing"].tolist() == [False, True])
        self.assertTrue(pd.isna(df.loc[1, "weather_temperature_c"]))

    def test_feature_columns_and_sun_present(self):
        df = scenario_weather(self.scen([us(2021, 8, 20, 15)]), self.hourly, self.points)
        self.assertTrue(set(FEATURES) <= set(df.columns))
        self.assertGreater(df.loc[0, SUN], 30)  # 11:00 local in August

    def test_gaps_in_hourly_series_are_rejected(self):
        with self.assertRaises(ValueError):
            hourly_frame(raw_response("boston", "2021-08-20 08:00", 8, [0] * 8, gap_at=3))

    def test_request_pins_model_and_utc(self):
        params = request_params(1.3, 103.8, "2021-08-17", "2021-08-18")
        self.assertEqual(params["models"], WEATHER_MODEL)
        self.assertEqual(params["timezone"], "GMT")
        self.assertEqual(set(params["hourly"].split(",")), set(HOURLY))


class SchemaTest(unittest.TestCase):
    def test_weather_fields_are_declared_pre_simulation_features(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is required")
        from tools.validate_scenario_subset import DEFAULT_SCHEMA, check_leakage, check_schema_structure, load_schema
        schema = load_schema(DEFAULT_SCHEMA)
        self.assertEqual(check_schema_structure(schema), [])
        self.assertEqual(check_leakage(schema), [])
        fields = {f["name"]: f for f in schema["groups"]["weather_pre_simulation"]["fields"]}
        expected = [c for c, _ in HOURLY.values()] + [PREV_3H, SUN] if pd is not None else list(fields)
        self.assertEqual(sorted(fields), sorted(expected))
        self.assertTrue(all(f["model_feature"] and f["available_before_simulation"] for f in fields.values()))


if __name__ == "__main__":
    unittest.main()
