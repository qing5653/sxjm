"""
Statistical analysis tools for mathematical modeling.

Provides descriptive statistics, hypothesis testing, confidence intervals,
and correlation analysis.
"""

import numpy as np
from scipy import stats


class Statistics:
    """Collection of statistical analysis utilities."""

    @staticmethod
    def describe(data):
        """Compute descriptive statistics for a dataset.

        Parameters
        ----------
        data : array-like
            1-D data array.

        Returns
        -------
        dict
            Keys: ``n``, ``mean``, ``median``, ``std``, ``variance``,
            ``min``, ``max``, ``range``, ``q1``, ``q3``, ``iqr``,
            ``skewness``, ``kurtosis``.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a 1-D array")
        q1, q3 = np.percentile(data, [25, 75])
        return {
            "n": len(data),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data, ddof=1)),
            "variance": float(np.var(data, ddof=1)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.max(data) - np.min(data)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
        }

    @staticmethod
    def t_test_one_sample(data, popmean):
        """One-sample t-test.

        Tests whether the sample mean differs significantly from
        *popmean*.

        Parameters
        ----------
        data : array-like
            Sample data.
        popmean : float
            Hypothesized population mean.

        Returns
        -------
        dict
            Keys: ``t_statistic``, ``p_value``, ``reject_h0`` (at α=0.05).
        """
        data = np.asarray(data, dtype=float)
        t_stat, p_value = stats.ttest_1samp(data, popmean)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "reject_h0": bool(p_value < 0.05),
        }

    @staticmethod
    def t_test_two_sample(data1, data2, equal_var=True):
        """Two-sample t-test.

        Tests whether the means of two independent samples differ
        significantly.

        Parameters
        ----------
        data1, data2 : array-like
            Sample data.
        equal_var : bool
            If ``True`` perform Student's t-test; if ``False`` perform
            Welch's t-test.

        Returns
        -------
        dict
            Keys: ``t_statistic``, ``p_value``, ``reject_h0`` (at α=0.05).
        """
        d1 = np.asarray(data1, dtype=float)
        d2 = np.asarray(data2, dtype=float)
        t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=equal_var)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "reject_h0": bool(p_value < 0.05),
        }

    @staticmethod
    def confidence_interval(data, confidence=0.95):
        """Compute the confidence interval for the mean.

        Parameters
        ----------
        data : array-like
            Sample data.
        confidence : float
            Confidence level (e.g. 0.95 for 95 %).

        Returns
        -------
        tuple of float
            ``(lower, upper)`` bounds of the confidence interval.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
        return (float(mean - margin), float(mean + margin))

    @staticmethod
    def correlation(x, y):
        """Pearson correlation coefficient and p-value.

        Parameters
        ----------
        x, y : array-like
            Data arrays of equal length.

        Returns
        -------
        dict
            Keys: ``r`` (correlation coefficient), ``p_value``.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r, p_value = stats.pearsonr(x, y)
        return {"r": float(r), "p_value": float(p_value)}

    @staticmethod
    def normality_test(data):
        """Shapiro-Wilk normality test.

        Parameters
        ----------
        data : array-like
            Sample data (3 ≤ n ≤ 5000).

        Returns
        -------
        dict
            Keys: ``statistic``, ``p_value``, ``is_normal`` (at α=0.05).
        """
        data = np.asarray(data, dtype=float)
        statistic, p_value = stats.shapiro(data)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": bool(p_value >= 0.05),
        }
