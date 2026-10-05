import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from src.car_model import CarModel


# --- Fixtures ---

@pytest.fixture
def viper_row():
    """Minimal row matching the Dodge Viper ACR-E schema."""
    return pd.Series({
        'year': 2016, 'make': 'Dodge', 'model': 'Viper', 'trim': 'ACR-E',
        'top_speed': 177,
        'accel_0_60': 3.3, 'accel_0_100': 7.4,
        'qmile_time': 11.3, 'qmile_speed': 127.3,
        'accel_0_186': None,
        'mass': 1538,
        'downforce_kg': 771, 'downforce_speed': 177,
        'horsepower': 645, 'torque': 600, 'layout': 'fr',
    })


@pytest.fixture
def veyron_row():
    return pd.Series({
        'year': 2010, 'make': 'Bugatti', 'model': 'Veyron', 'trim': 'Supersport',
        'top_speed': 267,
        'accel_0_60': 2.4, 'accel_0_100': 4.9,
        'qmile_time': 9.9, 'qmile_speed': 146,
        'accel_0_186': 14.6,
        'mass': 1990,
        'downforce_kg': 400, 'downforce_speed': 250,
        'horsepower': 1183, 'torque': 1106, 'layout': 'ma',
    })


@pytest.fixture
def minimal_row():
    """Only 0-60 data — the absolute minimum."""
    return pd.Series({
        'year': 2020, 'make': 'Test', 'model': 'Car', 'trim': 'Base',
        'top_speed': 120,
        'accel_0_60': 5.0,
        'accel_0_100': None, 'qmile_time': None, 'qmile_speed': None,
        'accel_0_186': None,
        'mass': 1000,
        'downforce_kg': 100, 'downforce_speed': 100,
        'horsepower': 200, 'torque': 200, 'layout': 'fr',
    })


# --- ID ---

class TestCarModelId:
    def test_id_format(self, viper_row):
        car = CarModel(viper_row)
        assert car.id == "2016_Dodge_Viper_ACR-E"

    def test_id_with_none_trim(self):
        row = pd.Series({
            'year': 2020, 'make': 'A', 'model': 'B', 'trim': None,
            'top_speed': 100, 'accel_0_60': 5.0,
            'accel_0_100': None, 'qmile_time': None, 'qmile_speed': None,
            'accel_0_186': None,
            'mass': 1000, 'downforce_kg': 100, 'downforce_speed': 100,
            'horsepower': 200, 'torque': 200, 'layout': 'fr',
        })
        car = CarModel(row)
        # trim is .fillna('') in the engine but not here — check actual output
        assert car.id.startswith("2020_A_B_")


# --- Acceleration points ---

class TestExtractAccelerationPoints:
    def test_all_points_present(self, viper_row):
        car = CarModel(viper_row)
        assert len(car.t_data) == 4  # origin + 3 data points
        assert car.t_data[0] == 0
        assert car.v_data[0] == 0

    def test_fewest_points(self, minimal_row):
        car = CarModel(minimal_row)
        assert len(car.t_data) == 2  # origin + 0-60

    def test_insufficient_points_raises(self):
        row = pd.Series({
            'year': 2020, 'make': 'X', 'model': 'Y', 'trim': 'Z',
            'top_speed': 100,
            'accel_0_60': None, 'accel_0_100': None,
            'qmile_time': None, 'qmile_speed': None,
            'accel_0_186': None,
            'mass': 1000, 'downforce_kg': 100, 'downforce_speed': 100,
            'horsepower': 200, 'torque': 200, 'layout': 'fr',
        })
        with pytest.raises(ValueError, match="insufficient acceleration points"):
            CarModel(row)

    def test_points_sorted_by_time(self, veyron_row):
        car = CarModel(veyron_row)
        assert list(car.t_data) == sorted(car.t_data)


# --- Acceleration curve ---

class TestAccelerationCurve:
    def test_curve_is_dataframe(self, viper_row):
        car = CarModel(viper_row)
        assert isinstance(car.acceleration_curve, pd.DataFrame)
        assert set(car.acceleration_curve.columns) == {'time', 'velocity'}

    def test_curve_starts_at_origin(self, viper_row):
        car = CarModel(viper_row)
        assert car.acceleration_curve['time'].iloc[0] == 0
        assert car.acceleration_curve['velocity'].iloc[0] == 0

    def test_curve_is_monotonically_increasing_velocity(self, viper_row):
        car = CarModel(viper_row)
        v = car.acceleration_curve['velocity']
        assert (v.diff().dropna() >= -1e-9).all()

    def test_curve_ends_at_top_speed(self, viper_row):
        car = CarModel(viper_row)
        final_v = car.acceleration_curve['velocity'].iloc[-1]
        expected = 177 * 0.44704
        assert abs(final_v - expected) < 0.5  # small tolerance for interpolation

    def test_curve_has_many_points(self, viper_row):
        car = CarModel(viper_row)
        assert len(car.acceleration_curve) > 1000


# --- Downforce curve ---

class TestDownforceCurve:
    def test_curve_is_dataframe(self, viper_row):
        car = CarModel(viper_row)
        assert isinstance(car.downforce_curve, pd.DataFrame)
        assert set(car.downforce_curve.columns) == {'velocity', 'downforce'}

    def test_downforce_starts_at_zero(self, viper_row):
        car = CarModel(viper_row)
        assert car.downforce_curve['downforce'].iloc[0] == pytest.approx(0, abs=1e-6)

    def test_downforce_increases_with_speed(self, viper_row):
        car = CarModel(viper_row)
        v = car.downforce_curve['downforce']
        assert (v.diff().dropna() >= -1e-9).all()

    def test_downforce_at_spec_speed(self, viper_row):
        car = CarModel(viper_row)
        v_at_speed = 177 * 0.44704
        # At downforce_speed, downforce should equal downforce_kg * 9.81
        expected = 771 * 9.81
        closest = car.downforce_curve.iloc[car.downforce_curve['velocity'].sub(v_at_speed).abs().idxmin()]
        assert closest['downforce'] == pytest.approx(expected, rel=0.05)


# --- Static helper ---

class TestMphToMps:
    def test_conversion(self):
        assert CarModel._mph_to_mps(60) == pytest.approx(60 * 0.44704)

    def test_zero(self):
        assert CarModel._mph_to_mps(0) == 0


# --- Edge cases ---

class TestEdgeCases:
    def test_qmile_none_but_others_present(self):
        row = pd.Series({
            'year': 2020, 'make': 'A', 'model': 'B', 'trim': 'C',
            'top_speed': 150,
            'accel_0_60': 4.0, 'accel_0_100': 8.0,
            'qmile_time': None, 'qmile_speed': None,
            'accel_0_186': None,
            'mass': 1200, 'downforce_kg': 200, 'downforce_speed': 150,
            'horsepower': 300, 'torque': 300, 'layout': 'fr',
        })
        car = CarModel(row)
        assert len(car.t_data) == 3  # origin + 0-60 + 0-100
