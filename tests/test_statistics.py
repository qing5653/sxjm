import math
import pytest
import numpy as np
from sxjm.statistics import Statistics


class TestDescribe:
    def test_basic(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        result = Statistics.describe(data)
        assert result["n"] == 8
        assert math.isclose(result["mean"], 5.0, rel_tol=1e-9)
        assert math.isclose(result["min"], 2.0)
        assert math.isclose(result["max"], 9.0)

    def test_std(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        result = Statistics.describe(data)
        expected_std = np.std(data, ddof=1)
        assert math.isclose(result["std"], expected_std, rel_tol=1e-9)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            Statistics.describe([[1, 2], [3, 4]])


class TestTTest:
    def test_one_sample_reject(self):
        rng = np.random.default_rng(0)
        data = rng.normal(10, 1, 100)
        result = Statistics.t_test_one_sample(data, popmean=0)
        assert result["reject_h0"] is True

    def test_one_sample_accept(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5, 1, 200)
        result = Statistics.t_test_one_sample(data, popmean=5)
        assert result["reject_h0"] is False

    def test_two_sample_reject(self):
        rng = np.random.default_rng(1)
        d1 = rng.normal(0, 1, 100)
        d2 = rng.normal(5, 1, 100)
        result = Statistics.t_test_two_sample(d1, d2)
        assert result["reject_h0"] is True

    def test_two_sample_accept(self):
        rng = np.random.default_rng(2)
        d1 = rng.normal(5, 1, 100)
        d2 = rng.normal(5, 1, 100)
        result = Statistics.t_test_two_sample(d1, d2)
        assert result["reject_h0"] is False


class TestConfidenceInterval:
    def test_contains_true_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, 500)
        lower, upper = Statistics.confidence_interval(data, confidence=0.95)
        assert lower < 10 < upper

    def test_wider_at_higher_confidence(self):
        data = np.arange(1, 101, dtype=float)
        lo95, hi95 = Statistics.confidence_interval(data, confidence=0.95)
        lo99, hi99 = Statistics.confidence_interval(data, confidence=0.99)
        assert (hi99 - lo99) > (hi95 - lo95)


class TestCorrelation:
    def test_perfect_positive(self):
        x = np.arange(10, dtype=float)
        y = 3 * x + 1
        result = Statistics.correlation(x, y)
        assert math.isclose(result["r"], 1.0, rel_tol=1e-9)

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        result = Statistics.correlation(x, y)
        assert abs(result["r"]) < 0.1


class TestNormalityTest:
    def test_normal_data(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, 100)
        result = Statistics.normality_test(data)
        assert result["is_normal"] is True

    def test_non_normal_data(self):
        rng = np.random.default_rng(4)
        data = rng.exponential(1, 500)
        result = Statistics.normality_test(data)
        assert result["is_normal"] is False
