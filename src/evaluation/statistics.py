"""Statistical analysis module for benchmark evaluation.

This module provides statistical analysis capabilities including:
- BCa (Bias-Corrected and Accelerated) bootstrap confidence intervals
- Paired sample comparisons with effect sizes
- Holm-Bonferroni correction for multiple comparison control

See ADR-003 for statistical significance as primary criterion.
See ADR-010 for Holm-Bonferroni multiple comparison correction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(slots=True, frozen=True)
class ConfidenceInterval:
    """Bootstrap confidence interval result.

    Attributes:
        mean: Point estimate (sample statistic)
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
        std_error: Standard error from bootstrap distribution
        n_iterations: Number of bootstrap iterations used
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    """

    mean: float
    lower: float
    upper: float
    std_error: float
    n_iterations: int
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate confidence interval bounds."""
        if self.lower > self.upper:
            raise ValueError(f"Lower bound ({self.lower}) cannot exceed upper bound ({self.upper})")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

    def contains(self, value: float) -> bool:
        """Check if a value falls within the confidence interval."""
        return self.lower <= value <= self.upper

    def width(self) -> float:
        """Return the width of the confidence interval."""
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "std_error": self.std_error,
            "n_iterations": self.n_iterations,
            "confidence_level": self.confidence_level,
        }


@dataclass(slots=True, frozen=True)
class ComparisonResult:
    """Statistical comparison between two conditions.

    Attributes:
        effect_size: Cohen's d effect size
        p_value: Raw p-value from statistical test
        p_value_corrected: P-value after Holm-Bonferroni correction
        is_significant: Whether the result is statistically significant
        ci_difference: Confidence interval on the difference
        comparison_name: Optional name for this comparison
    """

    effect_size: float
    p_value: float
    p_value_corrected: float
    is_significant: bool
    ci_difference: ConfidenceInterval
    comparison_name: str = ""

    def __post_init__(self) -> None:
        """Validate p-values."""
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be between 0 and 1, got {self.p_value}")
        if not 0.0 <= self.p_value_corrected <= 1.0:
            raise ValueError(
                f"p_value_corrected must be between 0 and 1, got {self.p_value_corrected}"
            )

    def effect_size_interpretation(self) -> str:
        """Interpret effect size using Cohen's conventions.

        Returns:
            Human-readable interpretation of effect size magnitude
        """
        d = abs(self.effect_size)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "p_value_corrected": self.p_value_corrected,
            "is_significant": self.is_significant,
            "ci_difference": self.ci_difference.to_dict(),
            "comparison_name": self.comparison_name,
            "effect_size_interpretation": self.effect_size_interpretation(),
        }


class StatisticalAnalyzer:
    """Statistical analysis with bootstrap CI and Holm-Bonferroni correction.

    Implements:
    - ADR-003: Statistical significance (p < 0.05) as primary criterion
    - ADR-010: Holm-Bonferroni for multiple comparison control

    Example:
        ```python
        analyzer = StatisticalAnalyzer(n_bootstrap=2000, random_seed=42)

        # Compute confidence interval
        scores = np.array([0.85, 0.87, 0.82, 0.89, 0.86])
        ci = analyzer.bootstrap_ci(scores)
        print(f"Mean: {ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]")

        # Compare two conditions
        memory_scores = np.array([0.85, 0.87, 0.82, 0.89, 0.86])
        baseline_scores = np.array([0.72, 0.75, 0.71, 0.74, 0.73])
        result = analyzer.paired_comparison(memory_scores, baseline_scores)
        print(f"Effect size: {result.effect_size:.3f}, p={result.p_value:.4f}")
        ```

    Attributes:
        n_bootstrap: Number of bootstrap iterations (default 2000)
        confidence_level: Confidence level for intervals (default 0.95)
        alpha: Significance threshold (default 0.05)
    """

    def __init__(
        self,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        random_seed: int | None = 42,
        alpha: float = 0.05,
    ) -> None:
        """Initialize the statistical analyzer.

        Args:
            n_bootstrap: Number of bootstrap iterations (recommended >= 2000)
            confidence_level: Confidence level for intervals (0.0-1.0)
            random_seed: Random seed for reproducibility (None for random)
            alpha: Significance threshold for hypothesis tests
        """
        if n_bootstrap < 100:
            raise ValueError(f"n_bootstrap should be at least 100, got {n_bootstrap}")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        self._n_bootstrap = n_bootstrap
        self._confidence = confidence_level
        self._alpha = alpha
        self._rng = np.random.default_rng(random_seed)

    @property
    def n_bootstrap(self) -> int:
        """Number of bootstrap iterations."""
        return self._n_bootstrap

    @property
    def confidence_level(self) -> float:
        """Confidence level for intervals."""
        return self._confidence

    @property
    def alpha(self) -> float:
        """Significance threshold."""
        return self._alpha

    def bootstrap_ci(
        self,
        data: NDArray[np.floating[Any]],
        statistic: Callable[[NDArray[np.floating[Any]]], float] = np.mean,
    ) -> ConfidenceInterval:
        """Compute BCa (Bias-Corrected and Accelerated) bootstrap CI.

        The BCa method corrects for bias and skewness in the bootstrap
        distribution, providing more accurate intervals than the simple
        percentile method, especially for small samples.

        Args:
            data: Sample data array
            statistic: Function to compute the statistic (default: mean)

        Returns:
            ConfidenceInterval with bounds and metadata

        Raises:
            ValueError: If data has fewer than 2 elements
        """
        data = np.asarray(data, dtype=np.float64)

        if len(data) < 2:
            raise ValueError("Data must have at least 2 elements for bootstrap CI")

        # Original statistic
        original_stat = float(statistic(data))

        # Bootstrap resampling
        bootstrap_stats = np.array(
            [
                statistic(self._rng.choice(data, size=len(data), replace=True))
                for _ in range(self._n_bootstrap)
            ]
        )

        # BCa correction
        alpha = 1 - self._confidence

        # Bias correction factor (z0)
        proportion_less: float = float(np.mean(bootstrap_stats < original_stat))
        # Handle edge cases where all bootstrap stats are >= or < original
        if proportion_less == 0.0:
            proportion_less = 1.0 / (2.0 * self._n_bootstrap)
        elif proportion_less == 1.0:
            proportion_less = 1.0 - 1.0 / (2.0 * self._n_bootstrap)

        z0 = float(stats.norm.ppf(proportion_less))

        # Acceleration factor (jackknife estimate)
        jackknife_stats = np.array([statistic(np.delete(data, i)) for i in range(len(data))])
        jackknife_mean = np.mean(jackknife_stats)
        diff_cubed = (jackknife_mean - jackknife_stats) ** 3
        diff_squared = (jackknife_mean - jackknife_stats) ** 2

        numerator = float(np.sum(diff_cubed))
        denominator = 6 * float(np.sum(diff_squared)) ** 1.5

        a = numerator / denominator if denominator != 0 else 0.0

        # Adjusted percentiles
        z_lower = float(stats.norm.ppf(alpha / 2))
        z_upper = float(stats.norm.ppf(1 - alpha / 2))

        # BCa adjusted percentiles
        def _bca_percentile(z: float) -> float:
            """Compute BCa-adjusted percentile."""
            denom = 1 - a * (z0 + z)
            if abs(denom) < 1e-10:
                # Avoid division by zero
                return 0.5
            return float(stats.norm.cdf(z0 + (z0 + z) / denom))

        lower_pct = _bca_percentile(z_lower)
        upper_pct = _bca_percentile(z_upper)

        # Clamp percentiles to valid range
        lower_pct = max(0.001, min(0.999, lower_pct))
        upper_pct = max(0.001, min(0.999, upper_pct))

        lower_bound = float(np.percentile(bootstrap_stats, lower_pct * 100))
        upper_bound = float(np.percentile(bootstrap_stats, upper_pct * 100))

        return ConfidenceInterval(
            mean=original_stat,
            lower=lower_bound,
            upper=upper_bound,
            std_error=float(np.std(bootstrap_stats, ddof=1)),
            n_iterations=self._n_bootstrap,
            confidence_level=self._confidence,
        )

    def paired_comparison(
        self,
        condition_a: NDArray[np.floating[Any]],
        condition_b: NDArray[np.floating[Any]],
        comparison_name: str = "",
    ) -> ComparisonResult:
        """Compare two conditions with paired test.

        Uses a paired t-test for significance and computes Cohen's d
        effect size for paired samples. The confidence interval is
        computed on the difference scores.

        Args:
            condition_a: Scores for condition A (e.g., memory system)
            condition_b: Scores for condition B (e.g., no-memory baseline)
            comparison_name: Optional name for this comparison

        Returns:
            ComparisonResult with effect size, p-value, and CI

        Raises:
            ValueError: If arrays have different lengths or fewer than 2 elements
        """
        condition_a = np.asarray(condition_a, dtype=np.float64)
        condition_b = np.asarray(condition_b, dtype=np.float64)

        if len(condition_a) != len(condition_b):
            raise ValueError(
                f"Arrays must have same length: {len(condition_a)} != {len(condition_b)}"
            )
        if len(condition_a) < 2:
            raise ValueError("Need at least 2 paired observations")

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(condition_a, condition_b)

        # Handle NaN p-value (e.g., when all differences are zero)
        if np.isnan(p_value):
            p_value = 1.0

        # Cohen's d for paired samples
        diff = condition_a - condition_b
        diff_std = float(np.std(diff, ddof=1))

        cohens_d = float(np.mean(diff)) / diff_std if diff_std > 0 else 0.0

        # Bootstrap CI on the difference
        diff_ci = self.bootstrap_ci(diff)

        return ComparisonResult(
            effect_size=cohens_d,
            p_value=float(p_value),
            p_value_corrected=float(p_value),  # Corrected in batch via holm_bonferroni
            is_significant=bool(p_value < self._alpha),
            ci_difference=diff_ci,
            comparison_name=comparison_name,
        )

    def holm_bonferroni_correction(
        self,
        p_values: list[float],
    ) -> list[float]:
        """Apply Holm-Bonferroni step-down correction.

        The Holm-Bonferroni method controls the family-wise error rate (FWER)
        while being more powerful than the simple Bonferroni correction. It
        works by sequentially testing p-values from smallest to largest.

        Implements ADR-010 for FWER control.

        Args:
            p_values: List of raw p-values from multiple comparisons

        Returns:
            List of corrected p-values (same order as input)

        Raises:
            ValueError: If any p-value is outside [0, 1]
        """
        if not p_values:
            return []

        # Validate p-values
        for i, p in enumerate(p_values):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p_value at index {i} must be between 0 and 1, got {p}")

        n = len(p_values)
        p_array = np.array(p_values)

        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Apply Holm-Bonferroni correction
        corrected = np.zeros(n)
        for rank, (orig_idx, p) in enumerate(zip(sorted_indices, sorted_p, strict=True)):
            # Multiply by (n - rank) where rank is 0-indexed
            corrected[orig_idx] = min(1.0, p * (n - rank))

        # Enforce monotonicity (corrected p-values must be non-decreasing)
        for i in range(1, n):
            idx_current = sorted_indices[i]
            idx_previous = sorted_indices[i - 1]
            corrected[idx_current] = max(corrected[idx_current], corrected[idx_previous])

        return corrected.tolist()

    def correct_comparisons(
        self,
        results: list[ComparisonResult],
    ) -> list[ComparisonResult]:
        """Apply Holm-Bonferroni correction to multiple comparison results.

        This method takes a list of ComparisonResults with raw p-values and
        returns new ComparisonResults with corrected p-values and updated
        significance flags.

        Args:
            results: List of ComparisonResult objects

        Returns:
            List of new ComparisonResult objects with corrected p-values
        """
        if not results:
            return []

        # Extract raw p-values
        raw_p_values = [r.p_value for r in results]

        # Apply correction
        corrected_p_values = self.holm_bonferroni_correction(raw_p_values)

        # Create new results with corrected values
        corrected_results = []
        for result, corrected_p in zip(results, corrected_p_values, strict=True):
            corrected_results.append(
                ComparisonResult(
                    effect_size=result.effect_size,
                    p_value=result.p_value,
                    p_value_corrected=corrected_p,
                    is_significant=bool(corrected_p < self._alpha),
                    ci_difference=result.ci_difference,
                    comparison_name=result.comparison_name,
                )
            )

        return corrected_results

    def summary_statistics(
        self,
        data: NDArray[np.floating[Any]],
    ) -> dict[str, float]:
        """Compute summary statistics for a dataset.

        Args:
            data: Array of values

        Returns:
            Dictionary with mean, std, min, max, median, and IQR
        """
        data = np.asarray(data, dtype=np.float64)

        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "n": len(data),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get analyzer configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            "n_bootstrap": self._n_bootstrap,
            "confidence_level": self._confidence,
            "alpha": self._alpha,
        }
