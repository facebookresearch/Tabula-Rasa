#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test normality of HDR image sequences using multiple statistical tests.

This script analyzes a sequence of OpenEXR HDR images to determine if the pixel
values are drawn from a normal distribution. It uses multiple approaches for
robust normality testing:

1. Shapiro-Wilk test (scipy.stats.shapiro) - Limited to 5000 samples, but very
   powerful for detecting non-normality. Uses bootstrap resampling for reliability.

2. D'Agostino-Pearson test (scipy.stats.normaltest) - No sample size limit,
   tests based on skewness and kurtosis.

3. Anderson-Darling test (scipy.stats.anderson) - No sample size limit,
   emphasis on tail behavior.

4. Descriptive statistics - Skewness and kurtosis computed on ALL pixels,
   providing distribution shape information regardless of sample size.

For reference on the Shapiro-Wilk test:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
- https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_shapiro.html

Usage:
    python test_normality.py --input <path_to_dir_or_zip> --output <output.csv> [options]

Examples:
    # Test images in a directory (uses all tests by default)
    python test_normality.py --input /path/to/images/ --output results.csv

    # Test images in a zip file
    python test_normality.py --input /path/to/images.zip --output results.csv --is-zip

    # Use bootstrap resampling for more reliable Shapiro-Wilk results
    python test_normality.py --input /path/to/images/ --output results.csv --bootstrap 10

    # Only run Shapiro-Wilk test (original behavior)
    python test_normality.py --input /path/to/images/ --output results.csv --test shapiro
"""

import argparse
import csv
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import Imath
import numpy as np
import OpenEXR
from scipy import stats


@dataclass
class NormalityResult:
    """Results from normality testing on a single image."""

    filename: str
    num_pixels: int
    num_valid_pixels: int

    # Shapiro-Wilk test results (on sampled data)
    shapiro_statistic: float
    shapiro_pvalue: float
    shapiro_pvalue_std: float  # Standard deviation from bootstrap (if used)

    # D'Agostino-Pearson test results (on all data)
    dagostino_statistic: float
    dagostino_pvalue: float

    # Anderson-Darling test results (on all data)
    anderson_statistic: float
    anderson_critical_5pct: float  # Critical value at 5% significance

    # Descriptive statistics (on all data)
    mean: float
    std: float
    skewness: float
    kurtosis: float  # Excess kurtosis (0 for normal distribution)


def read_exr_image(filepath: str) -> np.ndarray:
    """
    Read an OpenEXR image and return pixel values as a numpy array.

    Args:
        filepath: Path to the OpenEXR file.

    Returns:
        A 1D numpy array containing all pixel values (flattened across all channels).
    """
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()

    # Get data window dimensions
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Get available channels
    channels = list(header["channels"].keys())

    # Read all channels
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    all_values = []

    for channel in channels:
        channel_data = exr_file.channel(channel, pixel_type)
        channel_array = np.frombuffer(channel_data, dtype=np.float32)
        channel_array = channel_array.reshape((height, width))
        all_values.append(channel_array.flatten())

    exr_file.close()

    # Concatenate all channel values
    return np.concatenate(all_values)


def read_exr_from_bytes(data: bytes, temp_dir: str, filename: str) -> np.ndarray:
    """
    Read an OpenEXR image from bytes by writing to a temporary file.

    Args:
        data: Raw bytes of the EXR file.
        temp_dir: Temporary directory to write the file.
        filename: Name of the file.

    Returns:
        A 1D numpy array containing all pixel values.
    """
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, "wb") as f:
        f.write(data)
    result = read_exr_image(temp_path)
    os.remove(temp_path)
    return result


def shapiro_wilk_test_bootstrap(
    data: np.ndarray,
    sample_size: int = 5000,
    n_bootstrap: int = 1,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Perform Shapiro-Wilk test with optional bootstrap resampling for reliability.

    The Shapiro-Wilk test tests the null hypothesis that the data was drawn
    from a normal distribution. Since scipy limits samples to 5000, we use
    bootstrap resampling to get more reliable estimates from large datasets.

    Args:
        data: 1D array of values to test (can be much larger than 5000).
        sample_size: Number of samples per bootstrap iteration (max 5000).
        n_bootstrap: Number of bootstrap iterations. Higher values give more
                     reliable p-value estimates but take longer.
        random_seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (mean_statistic, mean_pvalue, pvalue_std).
        - mean_statistic: Average W statistic across bootstrap iterations.
        - mean_pvalue: Average p-value across bootstrap iterations.
        - pvalue_std: Standard deviation of p-values (0 if n_bootstrap=1).
    """
    # Remove NaN and Inf values
    valid_data = data[np.isfinite(data)]

    if len(valid_data) < 3:
        return np.nan, np.nan, np.nan

    # Shapiro-Wilk in scipy is limited to 5000 samples
    sample_size = min(sample_size, 5000)
    rng = np.random.default_rng(random_seed)

    statistics = []
    pvalues = []

    for _ in range(n_bootstrap):
        if len(valid_data) > sample_size:
            sample = rng.choice(valid_data, size=sample_size, replace=False)
        else:
            sample = valid_data

        result = stats.shapiro(sample)
        statistics.append(result.statistic)
        pvalues.append(result.pvalue)

    return np.mean(statistics), np.mean(pvalues), np.std(pvalues)


def dagostino_pearson_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Perform D'Agostino-Pearson test on the data.

    This test combines skewness and kurtosis to produce an omnibus test of
    normality. Unlike Shapiro-Wilk, it has no sample size limit and uses
    ALL the data.

    Args:
        data: 1D array of values to test.

    Returns:
        Tuple of (statistic, p-value).
    """
    valid_data = data[np.isfinite(data)]

    if len(valid_data) < 20:
        # D'Agostino-Pearson requires at least 20 samples
        return np.nan, np.nan

    result = stats.normaltest(valid_data)
    return result.statistic, result.pvalue


def anderson_darling_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Perform Anderson-Darling test on the data.

    This test gives more weight to the tails of the distribution compared
    to other tests. It has no sample size limit and uses ALL the data.

    Args:
        data: 1D array of values to test.

    Returns:
        Tuple of (statistic, critical_value_at_5pct).
        If statistic > critical_value_at_5pct, reject normality at 5% level.
    """
    valid_data = data[np.isfinite(data)]

    if len(valid_data) < 3:
        return np.nan, np.nan

    result = stats.anderson(valid_data, dist="norm")
    # Return statistic and critical value at 5% significance level (index 2)
    return result.statistic, result.critical_values[2]


def compute_descriptive_stats(
    data: np.ndarray,
) -> Tuple[float, float, float, float, int, int]:
    """
    Compute descriptive statistics on ALL data.

    These statistics provide distribution shape information that helps
    assess normality without sampling limitations.

    Args:
        data: 1D array of values.

    Returns:
        Tuple of (mean, std, skewness, excess_kurtosis, num_pixels, num_valid).
        - skewness: 0 for symmetric distributions (like normal).
        - excess_kurtosis: 0 for normal distribution, positive for heavy tails.
    """
    num_pixels = len(data)
    valid_data = data[np.isfinite(data)]
    num_valid = len(valid_data)

    if num_valid < 3:
        return np.nan, np.nan, np.nan, np.nan, num_pixels, num_valid

    mean = np.mean(valid_data)
    std = np.std(valid_data)
    skewness = stats.skew(valid_data)
    kurtosis = stats.kurtosis(valid_data)  # Excess kurtosis (Fisher's definition)

    return mean, std, skewness, kurtosis, num_pixels, num_valid


def analyze_image(
    pixel_values: np.ndarray,
    filename: str,
    tests: List[str],
    sample_size: int = 5000,
    n_bootstrap: int = 1,
    random_seed: int = 42,
) -> NormalityResult:
    """
    Perform comprehensive normality analysis on an image.

    Args:
        pixel_values: 1D array of all pixel values from the image.
        filename: Name of the file being analyzed.
        tests: List of tests to run ("shapiro", "dagostino", "anderson", "descriptive").
        sample_size: Sample size for Shapiro-Wilk test.
        n_bootstrap: Number of bootstrap iterations for Shapiro-Wilk.
        random_seed: Random seed for reproducibility.

    Returns:
        NormalityResult with all test results.
    """
    # Compute descriptive stats (always, as they're used for context)
    mean, std, skewness, kurtosis, num_pixels, num_valid = compute_descriptive_stats(
        pixel_values
    )

    # Initialize with NaN
    shapiro_stat, shapiro_pval, shapiro_std = np.nan, np.nan, np.nan
    dagostino_stat, dagostino_pval = np.nan, np.nan
    anderson_stat, anderson_crit = np.nan, np.nan

    if "shapiro" in tests or "all" in tests:
        shapiro_stat, shapiro_pval, shapiro_std = shapiro_wilk_test_bootstrap(
            pixel_values, sample_size, n_bootstrap, random_seed
        )

    if "dagostino" in tests or "all" in tests:
        dagostino_stat, dagostino_pval = dagostino_pearson_test(pixel_values)

    if "anderson" in tests or "all" in tests:
        anderson_stat, anderson_crit = anderson_darling_test(pixel_values)

    return NormalityResult(
        filename=filename,
        num_pixels=num_pixels,
        num_valid_pixels=num_valid,
        shapiro_statistic=shapiro_stat,
        shapiro_pvalue=shapiro_pval,
        shapiro_pvalue_std=shapiro_std,
        dagostino_statistic=dagostino_stat,
        dagostino_pvalue=dagostino_pval,
        anderson_statistic=anderson_stat,
        anderson_critical_5pct=anderson_crit,
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def get_exr_files_from_directory(directory: str) -> List[str]:
    """Get all EXR files from a directory, sorted by name."""
    exr_files = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith((".exr",)):
            exr_files.append(os.path.join(directory, filename))
    return exr_files


def get_exr_files_from_zip(zip_path: str) -> List[str]:
    """Get all EXR file names from a zip archive, sorted by name."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        exr_files = [
            name
            for name in sorted(zf.namelist())
            if name.lower().endswith((".exr",)) and not name.startswith("__MACOSX")
        ]
    return exr_files


def process_images(
    input_path: str,
    is_zip: bool,
    tests: List[str],
    sample_size: int = 5000,
    n_bootstrap: int = 1,
    verbose: bool = True,
) -> List[NormalityResult]:
    """
    Process all images and compute normality tests for each.

    Args:
        input_path: Path to directory or zip file containing EXR images.
        is_zip: Whether the input is a zip file.
        tests: List of tests to run.
        sample_size: Sample size for Shapiro-Wilk test.
        n_bootstrap: Number of bootstrap iterations for Shapiro-Wilk.
        verbose: Whether to print progress information.

    Returns:
        List of NormalityResult for each frame.
    """
    results = []

    if is_zip:
        exr_files = get_exr_files_from_zip(input_path)
        if verbose:
            print(f"Found {len(exr_files)} EXR files in zip archive")

        with zipfile.ZipFile(input_path, "r") as zf:
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, exr_name in enumerate(exr_files):
                    if verbose:
                        print(f"Processing [{i + 1}/{len(exr_files)}]: {exr_name}")

                    data = zf.read(exr_name)
                    basename = os.path.basename(exr_name)
                    pixel_values = read_exr_from_bytes(data, temp_dir, basename)

                    result = analyze_image(
                        pixel_values, exr_name, tests, sample_size, n_bootstrap
                    )
                    results.append(result)

                    if verbose:
                        _print_result_summary(result, tests)
    else:
        exr_files = get_exr_files_from_directory(input_path)
        if verbose:
            print(f"Found {len(exr_files)} EXR files in directory")

        for i, exr_path in enumerate(exr_files):
            filename = os.path.basename(exr_path)
            if verbose:
                print(f"Processing [{i + 1}/{len(exr_files)}]: {filename}")

            pixel_values = read_exr_image(exr_path)
            result = analyze_image(
                pixel_values, filename, tests, sample_size, n_bootstrap
            )
            results.append(result)

            if verbose:
                _print_result_summary(result, tests)

    return results


def _print_result_summary(result: NormalityResult, tests: List[str]) -> None:
    """Print a summary of the normality test results."""
    print(f"  Pixels: {result.num_valid_pixels:,} valid / {result.num_pixels:,} total")
    print(f"  Mean: {result.mean:.6f}, Std: {result.std:.6f}")

    if "shapiro" in tests or "all" in tests:
        if result.shapiro_pvalue_std > 0:
            print(
                f"  Shapiro-Wilk: W={result.shapiro_statistic:.6f}, "
                f"p={result.shapiro_pvalue:.6e} (±{result.shapiro_pvalue_std:.2e})"
            )
        else:
            print(
                f"  Shapiro-Wilk: W={result.shapiro_statistic:.6f}, "
                f"p={result.shapiro_pvalue:.6e}"
            )

    if "dagostino" in tests or "all" in tests:
        print(
            f"  D'Agostino-Pearson: k²={result.dagostino_statistic:.4f}, "
            f"p={result.dagostino_pvalue:.6e}"
        )

    if "anderson" in tests or "all" in tests:
        normal_str = (
            "NORMAL"
            if result.anderson_statistic < result.anderson_critical_5pct
            else "NOT NORMAL"
        )
        print(
            f"  Anderson-Darling: A²={result.anderson_statistic:.4f}, "
            f"crit(5%)={result.anderson_critical_5pct:.4f} [{normal_str}]"
        )

    if "descriptive" in tests or "all" in tests:
        print(
            f"  Descriptive: skew={result.skewness:.4f}, "
            f"kurtosis={result.kurtosis:.4f}"
        )


def save_results_to_csv(
    results: List[NormalityResult],
    output_path: str,
    tests: List[str],
    alpha: float = 0.05,
) -> None:
    """
    Save comprehensive results to a CSV file.

    Args:
        results: List of NormalityResult objects.
        output_path: Path to output CSV file.
        tests: List of tests that were run.
        alpha: Significance level for normality interpretation.
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Build header based on tests run
        # Always include mean and std for pixel intensities
        header = ["Frame", "Filename", "Total_Pixels", "Valid_Pixels", "Mean", "Std"]

        if "shapiro" in tests or "all" in tests:
            header.extend(
                ["Shapiro_W", "Shapiro_pvalue", "Shapiro_pvalue_std", "Shapiro_Normal"]
            )

        if "dagostino" in tests or "all" in tests:
            header.extend(["DAgostino_k2", "DAgostino_pvalue", "DAgostino_Normal"])

        if "anderson" in tests or "all" in tests:
            header.extend(["Anderson_A2", "Anderson_Crit_5pct", "Anderson_Normal"])

        if "descriptive" in tests or "all" in tests:
            header.extend(["Skewness", "Kurtosis"])

        writer.writerow(header)

        # Write per-frame results
        for i, r in enumerate(results):
            # Always include mean and std
            row = [
                i + 1,
                r.filename,
                r.num_pixels,
                r.num_valid_pixels,
                f"{r.mean:.6f}",
                f"{r.std:.6f}",
            ]

            if "shapiro" in tests or "all" in tests:
                is_normal = (
                    "Yes"
                    if (not np.isnan(r.shapiro_pvalue) and r.shapiro_pvalue >= alpha)
                    else "No"
                )
                row.extend(
                    [
                        f"{r.shapiro_statistic:.6f}",
                        f"{r.shapiro_pvalue:.6e}",
                        f"{r.shapiro_pvalue_std:.6e}",
                        is_normal,
                    ]
                )

            if "dagostino" in tests or "all" in tests:
                is_normal = (
                    "Yes"
                    if (
                        not np.isnan(r.dagostino_pvalue) and r.dagostino_pvalue >= alpha
                    )
                    else "No"
                )
                row.extend(
                    [
                        f"{r.dagostino_statistic:.4f}",
                        f"{r.dagostino_pvalue:.6e}",
                        is_normal,
                    ]
                )

            if "anderson" in tests or "all" in tests:
                is_normal = (
                    "Yes"
                    if (
                        not np.isnan(r.anderson_statistic)
                        and r.anderson_statistic < r.anderson_critical_5pct
                    )
                    else "No"
                )
                row.extend(
                    [
                        f"{r.anderson_statistic:.4f}",
                        f"{r.anderson_critical_5pct:.4f}",
                        is_normal,
                    ]
                )

            if "descriptive" in tests or "all" in tests:
                row.extend(
                    [
                        f"{r.skewness:.6f}",
                        f"{r.kurtosis:.6f}",
                    ]
                )

            writer.writerow(row)

        # Write summary statistics
        writer.writerow([])
        writer.writerow(["Summary Statistics"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Frames", len(results)])

        # Always report average mean and std of pixel intensities
        valid_mean = [r for r in results if not np.isnan(r.mean)]
        if valid_mean:
            avg_mean = np.mean([r.mean for r in valid_mean])
            avg_std = np.mean([r.std for r in valid_mean])
            std_of_means = np.std([r.mean for r in valid_mean])
            std_of_stds = np.std([r.std for r in valid_mean])
            writer.writerow(["Avg Pixel Intensity (Mean)", f"{avg_mean:.6f}"])
            writer.writerow(["Std of Pixel Intensity Means", f"{std_of_means:.6f}"])
            writer.writerow(["Avg Pixel Intensity (Std)", f"{avg_std:.6f}"])
            writer.writerow(["Std of Pixel Intensity Stds", f"{std_of_stds:.6f}"])

        if "shapiro" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.shapiro_pvalue)]
            if valid:
                avg_w = np.mean([r.shapiro_statistic for r in valid])
                avg_p = np.mean([r.shapiro_pvalue for r in valid])
                normal_count = sum(1 for r in valid if r.shapiro_pvalue >= alpha)
                writer.writerow(["Shapiro-Wilk Avg W", f"{avg_w:.6f}"])
                writer.writerow(["Shapiro-Wilk Avg p-value", f"{avg_p:.6e}"])
                writer.writerow(
                    [
                        f"Shapiro-Wilk Normal (p >= {alpha})",
                        f"{normal_count}/{len(valid)}",
                    ]
                )

        if "dagostino" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.dagostino_pvalue)]
            if valid:
                avg_p = np.mean([r.dagostino_pvalue for r in valid])
                normal_count = sum(1 for r in valid if r.dagostino_pvalue >= alpha)
                writer.writerow(["D'Agostino-Pearson Avg p-value", f"{avg_p:.6e}"])
                writer.writerow(
                    [
                        f"D'Agostino-Pearson Normal (p >= {alpha})",
                        f"{normal_count}/{len(valid)}",
                    ]
                )

        if "anderson" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.anderson_statistic)]
            if valid:
                avg_a = np.mean([r.anderson_statistic for r in valid])
                normal_count = sum(
                    1 for r in valid if r.anderson_statistic < r.anderson_critical_5pct
                )
                writer.writerow(["Anderson-Darling Avg A²", f"{avg_a:.4f}"])
                writer.writerow(
                    [
                        "Anderson-Darling Normal (A² < crit)",
                        f"{normal_count}/{len(valid)}",
                    ]
                )

        if "descriptive" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.skewness)]
            if valid:
                avg_skew = np.mean([r.skewness for r in valid])
                avg_kurt = np.mean([r.kurtosis for r in valid])
                writer.writerow(["Avg Skewness", f"{avg_skew:.6f}"])
                writer.writerow(["Avg Excess Kurtosis", f"{avg_kurt:.6f}"])

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test normality of HDR image sequences using multiple statistical tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input /path/to/images/ --output results.csv
  %(prog)s --input /path/to/images.zip --output results.csv --is-zip
  %(prog)s --input /path/to/images/ --output results.csv --bootstrap 10
  %(prog)s --input /path/to/images/ --output results.csv --test shapiro --test dagostino

Statistical Tests:
  - shapiro:     Shapiro-Wilk test (sampled, max 5000 pixels, very powerful)
  - dagostino:   D'Agostino-Pearson test (uses ALL pixels, tests skewness/kurtosis)
  - anderson:    Anderson-Darling test (uses ALL pixels, emphasizes tail behavior)
  - descriptive: Skewness and kurtosis statistics (uses ALL pixels)
  - all:         Run all tests (default)

Reliability Notes:
  The Shapiro-Wilk test is limited to 5000 samples by scipy. For images with
  millions of pixels, use --bootstrap N to run N independent samples and
  average the results for more reliable estimates. The D'Agostino-Pearson and
  Anderson-Darling tests have no sample size limits and use ALL pixel data.
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to directory or zip file containing EXR images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--is-zip",
        "-z",
        action="store_true",
        help="Specify if input is a zip file (auto-detected from .zip extension if not specified).",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="append",
        choices=["shapiro", "dagostino", "anderson", "descriptive", "all"],
        help="Which tests to run. Can be specified multiple times. Default: all.",
    )
    parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        default=5000,
        help="Sample size for Shapiro-Wilk test (default: 5000, max: 5000).",
    )
    parser.add_argument(
        "--bootstrap",
        "-b",
        type=int,
        default=1,
        help="Number of bootstrap iterations for Shapiro-Wilk (default: 1). "
        "Higher values give more reliable p-value estimates for large images.",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        help="Significance level for normality test interpretation (default: 0.05).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    # Set default tests if none specified
    tests = args.test if args.test else ["all"]

    # Auto-detect zip if not explicitly specified
    is_zip = args.is_zip or args.input.lower().endswith(".zip")

    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if is_zip and not zipfile.is_zipfile(args.input):
        print(f"Error: Input is not a valid zip file: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not is_zip and not os.path.isdir(args.input):
        print(f"Error: Input is not a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process images
    verbose = not args.quiet
    if verbose:
        print(f"Input: {args.input}")
        print(f"Input type: {'zip file' if is_zip else 'directory'}")
        print(f"Output: {args.output}")
        print(f"Tests: {', '.join(tests)}")
        if "shapiro" in tests or "all" in tests:
            print(f"Shapiro-Wilk sample size: {min(args.sample_size, 5000)}")
            print(f"Shapiro-Wilk bootstrap iterations: {args.bootstrap}")
        print(f"Significance level (alpha): {args.alpha}")
        print()

    results = process_images(
        args.input,
        is_zip,
        tests=tests,
        sample_size=args.sample_size,
        n_bootstrap=args.bootstrap,
        verbose=verbose,
    )

    if not results:
        print("Warning: No EXR files found in input.", file=sys.stderr)
        sys.exit(1)

    # Save results
    save_results_to_csv(results, args.output, tests=tests, alpha=args.alpha)

    # Print summary
    if verbose:
        print("\nSummary:")
        print(f"  Total frames analyzed: {len(results)}")

        # Always print average pixel intensity statistics
        valid_mean = [r for r in results if not np.isnan(r.mean)]
        if valid_mean:
            avg_mean = np.mean([r.mean for r in valid_mean])
            avg_std = np.mean([r.std for r in valid_mean])
            print(f"  Avg pixel intensity mean: {avg_mean:.6f}")
            print(f"  Avg pixel intensity std: {avg_std:.6f}")

        if "shapiro" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.shapiro_pvalue)]
            if valid:
                normal_count = sum(1 for r in valid if r.shapiro_pvalue >= args.alpha)
                print(
                    f"  Shapiro-Wilk: {normal_count}/{len(valid)} frames normal "
                    f"(p >= {args.alpha})"
                )

        if "dagostino" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.dagostino_pvalue)]
            if valid:
                normal_count = sum(1 for r in valid if r.dagostino_pvalue >= args.alpha)
                print(
                    f"  D'Agostino-Pearson: {normal_count}/{len(valid)} frames normal "
                    f"(p >= {args.alpha})"
                )

        if "anderson" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.anderson_statistic)]
            if valid:
                normal_count = sum(
                    1 for r in valid if r.anderson_statistic < r.anderson_critical_5pct
                )
                print(
                    f"  Anderson-Darling: {normal_count}/{len(valid)} frames normal "
                    f"(A² < critical)"
                )

        if "descriptive" in tests or "all" in tests:
            valid = [r for r in results if not np.isnan(r.skewness)]
            if valid:
                avg_skew = np.mean([r.skewness for r in valid])
                avg_kurt = np.mean([r.kurtosis for r in valid])
                print(f"  Avg skewness: {avg_skew:.4f} (0 = symmetric)")
                print(f"  Avg excess kurtosis: {avg_kurt:.4f} (0 = normal tails)")


if __name__ == "__main__":
    main()
