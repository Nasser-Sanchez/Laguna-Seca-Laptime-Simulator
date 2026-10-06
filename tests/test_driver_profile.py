# tests/test_driver_profile.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.driver_profile import DriverProfile, SkillLevel


# --- Params ---

class TestGetParams:
    def test_novice_params(self):
        d = DriverProfile(SkillLevel.NOVICE)
        assert (d.a1, d.b1, d.a2, d.b2) == (2, 4, 3, 3)

    def test_pro_params(self):
        d = DriverProfile(SkillLevel.PRO)
        assert (d.a1, d.b1, d.a2, d.b2) == (2, 0.5, 10000, 2)

    def test_max_params(self):
        d = DriverProfile(SkillLevel.MAX)
        assert (d.a1, d.b1, d.a2, d.b2) == (1, 1, 1, 1)


# --- Multiplier return type ---

class TestMultiplierReturnType:
    def test_max_returns_tuple(self):
        d = DriverProfile(SkillLevel.MAX)
        result = d.get_multiplier()
        assert isinstance(result, tuple)
        assert result == (1, 1)

    def test_non_max_returns_list(self):
        d = DriverProfile(SkillLevel.NOVICE)
        result = d.get_multiplier()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_pro_returns_list(self):
        d = DriverProfile(SkillLevel.PRO)
        result = d.get_multiplier()
        assert isinstance(result, list)


# --- Multiplier ranges ---

class TestMultiplierRanges:
    def test_cornering_multiplier_in_range(self):
        """Beta-sampled cornering multiplier should be in [0, 1]."""
        for _ in range(100):
            d = DriverProfile(SkillLevel.NOVICE)
            m = d.get_multiplier()
            assert 0 <= m[0] <= 1

    def test_straight_multiplier_positive(self):
        """Straight multiplier = 2 - beta_sample, so should be in [1, 2]."""
        for _ in range(100):
            d = DriverProfile(SkillLevel.NOVICE)
            m = d.get_multiplier()
            assert 1 <= m[1] <= 2

    def test_max_multiplier_always_one(self):
        """MAX skill always returns 1, 1 — no randomness."""
        d = DriverProfile(SkillLevel.MAX)
        for _ in range(50):
            assert d.get_multiplier() == (1, 1)


# --- Determinism for MAX ---

class TestDeterminism:
    def test_max_is_deterministic(self):
        d = DriverProfile(SkillLevel.MAX)
        results = [d.get_multiplier() for _ in range(10)]
        assert all(r == (1, 1) for r in results)

    def test_non_max_is_stochastic(self):
        """Non-MAX should produce some variance."""
        d = DriverProfile(SkillLevel.NOVICE)
        corners = [d.get_multiplier()[0] for _ in range(50)]
        assert len(set(corners)) > 1  # not all identical
