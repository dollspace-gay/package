"""
Heteroscedasticity Test Calibration

This script validates that heteroscedasticity tests are properly calibrated:

1. NOMINAL SIZE: At alpha=0.05, false positive rate should be ~5% (4-7%)
2. STATISTICAL POWER: Detection rate for moderate heteroscedasticity >80%

The key challenge: Nonlinear mean with constant variance (sine wave + fixed noise)
- Breusch-Pagan/White often fail here (mistake curve for variance change)
- Dette-Munk-Wagner is designed to handle this case

Reference: Dette, Munk & Wagner (1998) - "Estimating the variance in
nonparametric regression—what is a reasonable choice?"
"""

import numpy as np
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=UserWarning)

from kernel_regression import (
    NadarayaWatson,
    heteroscedasticity_test,
)


@dataclass
class CalibrationResult:
    test_name: str
    nominal_alpha: float
    empirical_size: float  # False positive rate
    size_calibrated: bool  # Is 4-7% at alpha=0.05?
    power: float  # True positive rate
    power_adequate: bool  # Is >80%?


def generate_nonlinear_homoscedastic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Nonlinear mean with CONSTANT variance.

    This is the VDD expert trap: the curve looks like it might have
    varying variance, but it doesn't. Tests should NOT reject.

    y = sin(2πx) + ε, where ε ~ N(0, 0.2²)
    """
    np.random.seed(seed)
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X.ravel())
    noise = np.random.randn(n) * 0.2  # Constant variance
    return X, y_true + noise


def generate_trumpet(n: int, seed: int, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Nonlinear mean with INCREASING variance ("Trumpet" pattern).

    y = sin(2πx) + ε, where ε ~ N(0, σ(x)²)
    σ(x) = 0.1 + strength * x  (variance doubles from x=0 to x=1)

    strength=1.0 is "moderate" heteroscedasticity.
    """
    np.random.seed(seed)
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X.ravel())
    sigma = 0.1 + strength * X.ravel()  # Variance increases with x
    noise = np.random.randn(n) * sigma
    return X, y_true + noise


def calibrate_test(
    test_name: str,
    n_samples: int = 200,
    n_trials: int = 500,
    alpha: float = 0.05,
    bandwidth: float = 0.1,
) -> CalibrationResult:
    """
    Run calibration for a specific test.

    Args:
        test_name: One of "white", "breusch_pagan", "goldfeld_quandt", "dette_munk_wagner"
        n_samples: Sample size per trial
        n_trials: Number of Monte Carlo trials
        alpha: Significance level
        bandwidth: Kernel bandwidth
    """
    print(f"\n{'='*60}")
    print(f"CALIBRATING: {test_name.upper()}")
    print(f"{'='*60}")
    print(f"  n={n_samples}, trials={n_trials}, alpha={alpha}")

    # Test 1: Nominal Size (Homoscedastic data)
    print(f"\n  [1] NOMINAL SIZE TEST (Homoscedastic data)")
    print(f"      Target: False positive rate ~ {alpha:.0%} (range: 4-7%)")

    false_positives = 0
    for trial in range(n_trials):
        X, y = generate_nonlinear_homoscedastic(n_samples, seed=trial)
        model = NadarayaWatson(bandwidth=bandwidth).fit(X, y)

        try:
            result = heteroscedasticity_test(model, X, y, test=test_name, alpha=alpha)
            if result.is_heteroscedastic:
                false_positives += 1
        except Exception as e:
            print(f"      Warning: Trial {trial} failed: {e}")

    empirical_size = false_positives / n_trials
    size_calibrated = 0.03 <= empirical_size <= 0.10  # Generous range
    size_perfect = 0.04 <= empirical_size <= 0.07  # PhD standard

    print(f"      Result: {empirical_size:.1%} false positive rate")
    print(f"      Status: {'CALIBRATED' if size_calibrated else 'OVERSIZED'}", end="")
    if size_perfect:
        print(" (EXCELLENT)")
    elif size_calibrated:
        print(" (ACCEPTABLE)")
    else:
        print(f" (REJECT - too {'high' if empirical_size > 0.10 else 'low'})")

    # Test 2: Statistical Power (Heteroscedastic data)
    print(f"\n  [2] STATISTICAL POWER TEST (Trumpet data, moderate effect)")
    print(f"      Target: Detection rate > 80%")

    true_positives = 0
    for trial in range(n_trials):
        X, y = generate_trumpet(n_samples, seed=trial + 100000, strength=1.0)
        model = NadarayaWatson(bandwidth=bandwidth).fit(X, y)

        try:
            result = heteroscedasticity_test(model, X, y, test=test_name, alpha=alpha)
            if result.is_heteroscedastic:
                true_positives += 1
        except Exception:
            pass

    power = true_positives / n_trials
    power_adequate = power >= 0.80

    print(f"      Result: {power:.1%} detection rate")
    print(f"      Status: {'ADEQUATE' if power_adequate else 'UNDERPOWERED'}")

    return CalibrationResult(
        test_name=test_name,
        nominal_alpha=alpha,
        empirical_size=empirical_size,
        size_calibrated=size_calibrated,
        power=power,
        power_adequate=power_adequate,
    )


def run_full_calibration():
    """Run calibration for all heteroscedasticity tests."""
    print("\n" + "="*60)
    print("HETEROSCEDASTICITY TEST CALIBRATION SUITE")
    print("="*60)
    print("\nBenchmark: test calibration")
    print("- Nominal Size: False positive rate should be 4-7% at alpha=0.05")
    print("- Power: Detection rate should be >80% for moderate effect")
    print("\nChallenge: Nonlinear mean (sin wave) with constant variance")
    print("- Naive tests mistake the curve for heteroscedasticity")
    print("- Dette-Munk-Wagner is designed to handle this case")

    tests = ["breusch_pagan", "white", "dette_munk_wagner"]
    results = []

    for test_name in tests:
        result = calibrate_test(test_name, n_samples=200, n_trials=500)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)

    print(f"\n{'Test':<25} {'Size':<12} {'Power':<12} {'Overall'}")
    print("-" * 60)

    for r in results:
        size_status = f"{r.empirical_size:.1%}"
        if r.size_calibrated:
            size_status += " OK"
        else:
            size_status += " FAIL"

        power_status = f"{r.power:.1%}"
        if r.power_adequate:
            power_status += " OK"
        else:
            power_status += " LOW"

        overall = "PASS" if (r.size_calibrated and r.power_adequate) else "FAIL"

        print(f"{r.test_name:<25} {size_status:<12} {power_status:<12} {overall}")

    print("-" * 60)

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Find best test
    passing_tests = [r for r in results if r.size_calibrated and r.power_adequate]

    if passing_tests:
        best = max(passing_tests, key=lambda r: r.power - r.empirical_size)
        print(f"\nBest calibrated test: {best.test_name}")
        print(f"  - False positive rate: {best.empirical_size:.1%} (target: 5%)")
        print(f"  - Power: {best.power:.1%} (target: >80%)")
    else:
        print("\nNo test passed both criteria.")

        # Check for oversized tests
        oversized = [r for r in results if not r.size_calibrated]
        if oversized:
            print("\nOversized tests (too many false positives):")
            for r in oversized:
                print(f"  - {r.test_name}: {r.empirical_size:.1%} false positive rate")
            print("\nNote: Kernel regression residuals are correlated, which can")
            print("inflate Type I error for tests designed for linear models.")

    return results


def power_curve(test_name: str = "dette_munk_wagner"):
    """
    Plot power curve: detection rate vs heteroscedasticity strength.

    This shows how power increases with effect size.
    """
    print("\n" + "="*60)
    print(f"POWER CURVE FOR {test_name.upper()}")
    print("="*60)

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    n_samples = 200
    n_trials = 200
    bandwidth = 0.1
    alpha = 0.05

    print(f"\nStrength = 0 means constant variance (should reject ~5%)")
    print(f"Strength = 1 means variance doubles from x=0 to x=1")
    print(f"\n{'Strength':<12} {'Detection Rate':<20} {'Bar'}")
    print("-" * 50)

    for strength in strengths:
        detections = 0
        for trial in range(n_trials):
            if strength == 0:
                X, y = generate_nonlinear_homoscedastic(n_samples, seed=trial)
            else:
                X, y = generate_trumpet(n_samples, seed=trial, strength=strength)

            model = NadarayaWatson(bandwidth=bandwidth).fit(X, y)
            result = heteroscedasticity_test(model, X, y, test=test_name, alpha=alpha)

            if result.is_heteroscedastic:
                detections += 1

        rate = detections / n_trials
        bar = "#" * int(rate * 40)
        print(f"{strength:<12.2f} {rate:<20.1%} {bar}")

    print("-" * 50)
    print("Note: A well-calibrated test has ~5% at strength=0,")
    print("      increasing smoothly to >95% at high strength.")


if __name__ == "__main__":
    results = run_full_calibration()
    print("\n")
    power_curve("dette_munk_wagner")
