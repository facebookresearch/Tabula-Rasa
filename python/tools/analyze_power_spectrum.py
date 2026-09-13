#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Power Spectrum Analysis for Image Sequences

This script analyzes the power spectrum of multiple image sequences stored as
zip archives containing HDR frames in OpenEXR format. Each sequence is
interpreted as a single realization of a time-varying noise field.

The PSD estimation uses Welch's method with configurable:
- Patch size for local spectral analysis
- Overlap between patches
- Detrending (none, constant, linear)
- 2D Hann windowing to reduce spectral leakage

Usage:
    python analyze_power_spectrum.py archive1.zip archive2.zip ... -o output_dir
"""

import argparse
import csv
import glob
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import Imath
    import OpenEXR
except ImportError:
    raise ImportError("OpenEXR library required. Install with: pip install OpenEXR")


@dataclass
class WelchConfig:
    """Configuration for Welch's method PSD estimation."""

    patch_size: Optional[int] = None  # None means use full image
    overlap: float = 0.5  # Overlap fraction (0.0 to 1.0)
    detrend: str = "none"  # 'none', 'constant', 'linear'
    use_window: bool = True  # Apply 2D Hann window


def create_2d_hann_window(height: int, width: int) -> np.ndarray:
    """
    Create a 2D Hann window for reducing spectral leakage.

    Args:
        height: Window height
        width: Window width

    Returns:
        2D Hann window array of shape (height, width)
    """
    hann_y = np.hanning(height)
    hann_x = np.hanning(width)
    return np.outer(hann_y, hann_x)


def detrend_patch(patch: np.ndarray, method: str) -> np.ndarray:
    """
    Remove trend from a 2D patch.

    Args:
        patch: 2D input patch
        method: Detrending method ('none', 'constant', 'linear')

    Returns:
        Detrended patch
    """
    if method == "none":
        return patch
    elif method == "constant":
        # Remove mean (DC component)
        return patch - np.mean(patch)
    elif method == "linear":
        # Remove linear trend using least squares fit
        h, w = patch.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Flatten for linear regression
        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()
        z_flat = patch.ravel()

        # Build design matrix for plane fit: z = a*x + b*y + c
        A = np.column_stack([x_flat, y_flat, np.ones_like(x_flat)])

        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)

        # Compute and subtract the fitted plane
        plane = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]
        return patch - plane
    else:
        raise ValueError(f"Unknown detrend method: {method}")


def extract_patches_2d(
    image: np.ndarray, patch_size: int, overlap: float
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract overlapping patches from a 2D image.

    Args:
        image: 2D input image
        patch_size: Size of square patches
        overlap: Overlap fraction between patches (0.0 to 1.0)

    Returns:
        List of tuples (patch, row_start, col_start)
    """
    h, w = image.shape
    step = int(patch_size * (1 - overlap))
    step = max(1, step)  # Ensure at least 1 pixel step

    patches = []

    row = 0
    while row + patch_size <= h:
        col = 0
        while col + patch_size <= w:
            patch = image[row : row + patch_size, col : col + patch_size]
            patches.append((patch, row, col))
            col += step
        row += step

    return patches


def compute_welch_psd_2d(
    image: np.ndarray, config: WelchConfig
) -> Tuple[np.ndarray, int]:
    """
    Compute power spectral density using Welch's method for 2D images.

    Welch's method:
    1. Divide image into overlapping patches
    2. Optionally detrend each patch
    3. Apply window function to each patch
    4. Compute modified periodogram for each patch
    5. Average the periodograms

    Args:
        image: 2D input image
        config: Welch configuration parameters

    Returns:
        Tuple of (averaged PSD of shape (patch_size, patch_size), number of patches)
    """
    h, w = image.shape

    # Determine patch size
    if config.patch_size is None:
        # Use full image (standard periodogram)
        patch_size = min(h, w)
    else:
        patch_size = min(config.patch_size, h, w)

    # If patch size equals image size and no windowing, use simple periodogram
    if patch_size >= min(h, w) and not config.use_window and config.detrend == "none":
        # Crop to square if necessary
        size = min(h, w)
        cropped = image[:size, :size]
        fft = np.fft.fft2(cropped)
        fft_shifted = np.fft.fftshift(fft)
        psd = np.abs(fft_shifted) ** 2 / cropped.size
        return psd, 1

    # Create window if needed
    if config.use_window:
        window = create_2d_hann_window(patch_size, patch_size)
        # Compute window power for normalization (to preserve variance)
        window_power = np.mean(window**2)
    else:
        window = np.ones((patch_size, patch_size))
        window_power = 1.0

    # Extract patches
    patches = extract_patches_2d(image, patch_size, config.overlap)

    if not patches:
        # Image too small for even one patch, use what we can
        size = min(h, w)
        cropped = image[:size, :size]
        if config.detrend != "none":
            cropped = detrend_patch(cropped, config.detrend)
        if config.use_window:
            win = create_2d_hann_window(size, size)
            cropped = cropped * win
            window_power = np.mean(win**2)
        else:
            window_power = 1.0
        fft = np.fft.fft2(cropped)
        fft_shifted = np.fft.fftshift(fft)
        psd = np.abs(fft_shifted) ** 2 / (cropped.size * window_power)
        return psd, 1

    # Compute modified periodogram for each patch and accumulate
    psd_sum = np.zeros((patch_size, patch_size), dtype=np.float64)

    for patch, _, _ in patches:
        # Detrend
        if config.detrend != "none":
            patch = detrend_patch(patch, config.detrend)

        # Apply window
        windowed_patch = patch * window

        # Compute periodogram
        fft = np.fft.fft2(windowed_patch)
        fft_shifted = np.fft.fftshift(fft)

        # Normalize by number of points and window power
        periodogram = np.abs(fft_shifted) ** 2 / (
            patch_size * patch_size * window_power
        )
        psd_sum += periodogram

    # Average over patches
    avg_psd = psd_sum / len(patches)

    return avg_psd, len(patches)


def read_exr(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Read an OpenEXR file and return as numpy array.

    Args:
        filepath: Path to the EXR file

    Returns:
        Tuple of (numpy array of shape (H, W, C) with float32 values, channel names)
    """
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Get channel names - sort to ensure consistent ordering
    channels = sorted(header["channels"].keys())

    # Read channels
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_data = []

    for channel in channels:
        raw_data = exr_file.channel(channel, pt)
        channel_array = np.frombuffer(raw_data, dtype=np.float32).copy()
        channel_array = channel_array.reshape((height, width))
        channel_data.append(channel_array)

    exr_file.close()

    # Stack channels
    if len(channel_data) == 1:
        return channel_data[0], channels
    else:
        return np.stack(channel_data, axis=-1), channels


def write_exr(
    filepath: str, data: np.ndarray, channel_names: Optional[List[str]] = None
) -> None:
    """
    Write numpy array to OpenEXR file.

    Args:
        filepath: Output path
        data: numpy array of shape (H, W) or (H, W, C)
        channel_names: Optional list of channel names
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    height, width, num_channels = data.shape

    if channel_names is None:
        if num_channels == 1:
            channel_names = ["Y"]
        elif num_channels == 3:
            channel_names = ["R", "G", "B"]
        elif num_channels == 4:
            channel_names = ["R", "G", "B", "A"]
        else:
            channel_names = [f"C{i}" for i in range(num_channels)]

    header = OpenEXR.Header(width, height)
    header["channels"] = {
        name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        for name in channel_names
    }

    exr_file = OpenEXR.OutputFile(filepath, header)

    channel_dict = {}
    for i, name in enumerate(channel_names):
        channel_dict[name] = data[:, :, i].astype(np.float32).tobytes()

    exr_file.writePixels(channel_dict)
    exr_file.close()


def compute_power_spectrum_welch(
    image: np.ndarray, config: WelchConfig
) -> Tuple[np.ndarray, int]:
    """
    Compute the power spectrum using Welch's method for single or multi-channel images.

    Args:
        image: Input image array of shape (H, W) or (H, W, C)
        config: Welch configuration parameters

    Returns:
        Tuple of (power spectrum array, total number of patches used)
    """
    if image.ndim == 2:
        return compute_welch_psd_2d(image, config)
    else:
        # Multiple channels - process each separately
        power_spectra = []
        total_patches = 0
        for c in range(image.shape[-1]):
            psd, num_patches = compute_welch_psd_2d(image[:, :, c], config)
            power_spectra.append(psd)
            total_patches = num_patches  # Same for all channels
        return np.stack(power_spectra, axis=-1), total_patches


def compute_l2_error(psd: np.ndarray, target_psd: float = 1.0) -> float:
    """
    Compute the L2 error (RMSE) between the PSD and a constant target PSD.

    For ideal Gaussian white noise with unit variance, the PSD should be
    constant at 1.0 across all frequencies.

    Args:
        psd: Power spectrum array
        target_psd: Target constant PSD value (default 1.0 for unit variance white noise)

    Returns:
        L2 error (Mean Square Error)
    """
    error = np.mean((psd - target_psd) ** 2)
    return float(error)


def extract_frame_name(filename: str) -> str:
    """
    Extract a normalized frame name from a filename.

    Handles various naming conventions like:
        - frame_001.exr -> frame_1
        - frame_1.exr -> frame_1
        - 001.exr -> frame_1

    Args:
        filename: Input filename

    Returns:
        Normalized frame identifier
    """
    # Remove directory components and extension
    base = os.path.splitext(os.path.basename(filename))[0]

    # Try to extract frame number pattern
    # Pattern: prefix_number or just number
    match = re.match(r"^(.+?)_?(\d+)$", base)
    if match:
        prefix = match.group(1).rstrip("_")
        number = int(match.group(2))
        if prefix:
            return f"{prefix}_{number}"
        else:
            return f"frame_{number}"

    # If just a number
    if base.isdigit():
        return f"frame_{int(base)}"

    return base


def frame_sort_key(frame_name: str) -> Tuple:
    """
    Generate a sort key for frame names to ensure natural ordering.

    Args:
        frame_name: Frame name string

    Returns:
        Tuple for sorting
    """
    match = re.search(r"(\d+)", frame_name)
    if match:
        prefix = frame_name[: match.start()]
        number = int(match.group(1))
        suffix = frame_name[match.end() :]
        return (prefix, number, suffix)
    return (frame_name, 0, "")


def process_zip_archive(zip_path: str, temp_dir: str) -> Dict[str, str]:
    """
    Extract EXR files from a zip archive.

    Args:
        zip_path: Path to the zip archive
        temp_dir: Temporary directory for extraction

    Returns:
        Dictionary mapping frame names to extracted file paths
    """
    frames = {}
    archive_name = os.path.splitext(os.path.basename(zip_path))[0]
    extract_dir = os.path.join(temp_dir, archive_name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Skip directories and non-EXR files
            if name.endswith("/") or not name.lower().endswith(".exr"):
                continue

            # Extract file
            zf.extract(name, extract_dir)
            extracted_path = os.path.join(extract_dir, name)

            # Get normalized frame name
            frame_name = extract_frame_name(name)

            if frame_name in frames:
                print(f"  WARNING: Duplicate frame name '{frame_name}' in {zip_path}")

            frames[frame_name] = extracted_path

    return frames


def analyze_sequences(
    zip_paths: List[str],
    output_dir: str,
    target_variance: float = 1.0,
    welch_config: Optional[WelchConfig] = None,
) -> None:
    """
    Analyze power spectrum of multiple image sequences using Welch's method.

    Each sequence is treated as a single realization of a time-varying noise field.
    Frames with matching names are grouped together, and their power spectra are
    averaged to estimate the expected PSD.

    Args:
        zip_paths: List of paths to zip archives containing image sequences
        output_dir: Directory for output files
        target_variance: Target variance for ideal white noise (default 1.0)
        welch_config: Configuration for Welch's method (default: standard periodogram)
    """
    if welch_config is None:
        welch_config = WelchConfig(
            patch_size=None, overlap=0.5, detrend="none", use_window=False
        )

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract all archives and collect frames
        print("=" * 60)
        print("Extracting archives...")
        print("=" * 60)

        all_sequences: Dict[str, Dict[str, str]] = {}
        for zip_path in zip_paths:
            print(f"\nProcessing: {zip_path}")
            frames = process_zip_archive(zip_path, temp_dir)
            archive_name = os.path.splitext(os.path.basename(zip_path))[0]
            all_sequences[archive_name] = frames
            print(f"  Found {len(frames)} EXR frames")
            if frames:
                sample_frames = sorted(frames.keys(), key=frame_sort_key)[:5]
                print(f"  Sample frames: {sample_frames}")

        if not all_sequences:
            print("ERROR: No sequences found!")
            return

        # Find common frame names across all sequences
        print("\n" + "=" * 60)
        print("Analyzing frame groups...")
        print("=" * 60)

        all_frame_sets = [set(seq.keys()) for seq in all_sequences.values()]
        common_frames = set.intersection(*all_frame_sets) if all_frame_sets else set()
        all_frames = set.union(*all_frame_sets) if all_frame_sets else set()

        # Warn about mismatched frames
        mismatched_frames = all_frames - common_frames
        if mismatched_frames:
            print(
                f"\nWARNING: Found {len(mismatched_frames)} frames not present in all sequences:"
            )
            for archive_name, frames in all_sequences.items():
                archive_only = set(frames.keys()) - common_frames
                if archive_only:
                    print(
                        f"  '{archive_name}' has unique frames: {sorted(archive_only, key=frame_sort_key)}"
                    )

        if not common_frames:
            print("\nERROR: No common frames found across all sequences!")
            print("Available frames per archive:")
            for archive_name, frames in all_sequences.items():
                print(f"  {archive_name}: {sorted(frames.keys(), key=frame_sort_key)}")
            return

        print(
            f"\nFound {len(common_frames)} common frames across {len(all_sequences)} sequences"
        )

        # Print Welch configuration
        print("\n" + "=" * 60)
        print("Welch's Method Configuration")
        print("=" * 60)
        print(f"  Patch size: {welch_config.patch_size or 'full image'}")
        print(f"  Overlap: {welch_config.overlap:.0%}")
        print(f"  Detrend: {welch_config.detrend}")
        print(
            f"  Window: {'Hann' if welch_config.use_window else 'None (rectangular)'}"
        )

        # Process each frame group
        results = []
        sorted_frames = sorted(common_frames, key=frame_sort_key)
        channel_names = None

        print("\n" + "=" * 60)
        print("Computing power spectra...")
        print("=" * 60)

        for i, frame_name in enumerate(sorted_frames):
            print(
                f"\n[{i + 1}/{len(sorted_frames)}] Processing frame group: {frame_name}"
            )

            # Load all frames for this group
            frame_images = []
            for _archive_name, frames in all_sequences.items():
                img, channels = read_exr(frames[frame_name])
                frame_images.append(img)
                if channel_names is None:
                    channel_names = channels

            # Check shape consistency
            shapes = [img.shape for img in frame_images]
            if len(set(shapes)) > 1:
                print("  WARNING: Inconsistent image shapes in frame group:")
                for (arch_name, _), shape in zip(all_sequences.items(), shapes):
                    print(f"    {arch_name}: {shape}")
                # Use the minimum common shape
                min_shape = tuple(
                    min(s[i] for s in shapes) for i in range(len(shapes[0]))
                )
                frame_images = [
                    img[: min_shape[0], : min_shape[1]] for img in frame_images
                ]

            shape = frame_images[0].shape
            print("  Image shape: {}, Channels: {}".format(shape, channel_names))

            # Compute power spectrum for each frame realization using Welch's method
            power_spectra = []
            for img in frame_images:
                psd, num_patches = compute_power_spectrum_welch(img, welch_config)
                power_spectra.append(psd)
                total_patches = num_patches

            print(f"  Patches per frame: {total_patches}")

            # Average power spectra across realizations (further variance reduction)
            avg_psd = np.mean(power_spectra, axis=0)

            # Compute L2 error against target PSD (constant for white noise)
            l2_error = compute_l2_error(avg_psd, target_variance)

            # Also compute per-channel L2 errors for reporting
            if avg_psd.ndim == 3:
                channel_errors = [
                    compute_l2_error(avg_psd[:, :, c], target_variance)
                    for c in range(avg_psd.shape[-1])
                ]
            else:
                channel_errors = [l2_error]

            # Save average PSD image
            psd_output_path = os.path.join(output_dir, f"{frame_name}_avg_psd.exr")
            write_exr(psd_output_path, avg_psd, channel_names)
            print(f"  Saved: {psd_output_path}")
            print(f"  L2 error (overall): {l2_error:.6f}")
            if len(channel_errors) > 1:
                for ch_name, ch_err in zip(channel_names, channel_errors):
                    print(f"    Channel {ch_name}: {ch_err:.6f}")

            # Determine PSD dimensions
            psd_shape = avg_psd.shape

            # Store results
            result = {
                "frame_name": frame_name,
                "l2_error": l2_error,
                "num_sequences": len(frame_images),
                "num_patches": total_patches,
                "image_height": shape[0],
                "image_width": shape[1],
                "psd_height": psd_shape[0],
                "psd_width": psd_shape[1] if len(psd_shape) > 1 else psd_shape[0],
                "num_channels": shape[2] if len(shape) > 2 else 1,
            }
            # Add per-channel errors
            for ch_name, ch_err in zip(channel_names, channel_errors):
                result[f"l2_error_{ch_name}"] = ch_err

            results.append(result)

        # Write CSV report with per-frame results and summary statistics
        print("\n" + "=" * 60)
        print("Writing results...")
        print("=" * 60)

        csv_path = os.path.join(output_dir, "psd_analysis_results.csv")
        if results:
            l2_errors = [r["l2_error"] for r in results]

            # Compute summary statistics
            summary_stats = {
                "num_frame_groups": len(results),
                "num_sequences_per_group": results[0]["num_sequences"],
                "num_patches_per_frame": results[0]["num_patches"],
                "target_psd": target_variance,
                "patch_size": welch_config.patch_size or "full_image",
                "overlap": welch_config.overlap,
                "detrend": welch_config.detrend,
                "window": "hann" if welch_config.use_window else "rectangular",
                "l2_error_mean": float(np.mean(l2_errors)),
                "l2_error_std": float(np.std(l2_errors)),
                "l2_error_min": float(np.min(l2_errors)),
                "l2_error_max": float(np.max(l2_errors)),
                "l2_error_median": float(np.median(l2_errors)),
            }

            # Write combined CSV with per-frame results and summary
            fieldnames = list(results[0].keys())
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write per-frame results section
                csvfile.write("# Per-Frame Results\n")
                writer.writeheader()
                writer.writerows(results)

                # Write summary statistics section
                csvfile.write("\n# Summary Statistics\n")
                for key, value in summary_stats.items():
                    csvfile.write(f"{key},{value}\n")

            print(f"\nResults saved to: {csv_path}")

            # Print summary statistics to console
            print("\n" + "=" * 60)
            print("Summary Statistics")
            print("=" * 60)
            print(f"  Number of frame groups: {summary_stats['num_frame_groups']}")
            print(f"  Sequences per group: {summary_stats['num_sequences_per_group']}")
            print(f"  Patches per frame: {summary_stats['num_patches_per_frame']}")
            print("  Target PSD (white noise): {}".format(target_variance))
            print("\n  L2 Error Statistics:")
            print(f"    Mean:   {summary_stats['l2_error_mean']:.6f}")
            print(f"    Std:    {summary_stats['l2_error_std']:.6f}")
            print(f"    Min:    {summary_stats['l2_error_min']:.6f}")
            print(f"    Max:    {summary_stats['l2_error_max']:.6f}")
            print(f"    Median: {summary_stats['l2_error_median']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze power spectrum of image sequences stored in zip archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with standard periodogram
  %(prog)s seq1.zip seq2.zip seq3.zip -o results

  # Using Welch's method with 256x256 patches and 50%% overlap
  %(prog)s "data/*.zip" --patch-size 256 --overlap 0.5 -o analysis

  # Welch's method with detrending and Hann window (default)
  %(prog)s "data/*.zip" --patch-size 128 --detrend constant -o analysis

  # Disable windowing for rectangular window
  %(prog)s "data/*.zip" --patch-size 256 --no-window -o analysis

Glob patterns are supported and will be expanded by the script.
Quote patterns to prevent shell expansion (e.g., "data/*.zip").

Each zip archive should contain HDR frames in OpenEXR format with matching
frame names (e.g., frame_1.exr, frame_2.exr, ...).

Welch's Method Options:
  The PSD is estimated using Welch's method which divides each image into
  overlapping patches, computes a modified periodogram for each patch, and
  averages them for reduced variance. Options include:

  --patch-size    Size of square patches (default: full image = standard periodogram)
  --overlap       Fraction of overlap between patches, 0.0 to 1.0 (default: 0.5)
  --detrend       Remove trend from each patch: none, constant (mean), or linear
  --no-window     Disable 2D Hann window (uses rectangular window instead)

The script computes the power spectrum for each frame, averages across
sequences for each frame group, and compares against ideal Gaussian white
noise (constant PSD = target variance).
        """,
    )
    parser.add_argument(
        "archives",
        nargs="+",
        help="Paths or glob patterns to zip archives (e.g., 'data/*.zip')",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="psd_analysis",
        help="Output directory for results (default: psd_analysis)",
    )
    parser.add_argument(
        "--target-variance",
        type=float,
        default=1.0,
        help="Target variance for ideal white noise PSD (default: 1.0)",
    )

    # Welch's method arguments
    welch_group = parser.add_argument_group("Welch's method options")
    welch_group.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Size of square patches for Welch's method. If not specified, "
        "uses the full image (standard periodogram).",
    )
    welch_group.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between patches, 0.0 to 1.0 (default: 0.5)",
    )
    welch_group.add_argument(
        "--detrend",
        choices=["none", "constant", "linear"],
        default="none",
        help="Detrending method: 'none' (no detrending), 'constant' (remove mean), "
        "'linear' (remove linear trend). Default: none",
    )
    welch_group.add_argument(
        "--no-window",
        action="store_true",
        help="Disable 2D Hann window (use rectangular window instead). "
        "By default, Hann window is applied when using patches.",
    )

    args = parser.parse_args()

    # Validate overlap
    if not 0.0 <= args.overlap < 1.0:
        print("ERROR: Overlap must be in range [0.0, 1.0)")
        return 1

    # Create Welch configuration
    # Use window by default when patch_size is specified, disable if --no-window
    use_window = args.patch_size is not None and not args.no_window
    welch_config = WelchConfig(
        patch_size=args.patch_size,
        overlap=args.overlap,
        detrend=args.detrend,
        use_window=use_window,
    )

    # Expand glob patterns and validate inputs
    valid_archives = []
    for pattern in args.archives:
        # Try glob expansion first
        expanded_paths = glob.glob(pattern, recursive=True)

        if not expanded_paths:
            # No glob matches - treat as literal path
            if os.path.exists(pattern):
                expanded_paths = [pattern]
            else:
                print(f"WARNING: No matches found for pattern: {pattern}")
                continue

        for path in sorted(expanded_paths):
            if not zipfile.is_zipfile(path):
                print(f"WARNING: Not a valid zip file, skipping: {path}")
                continue
            valid_archives.append(path)

    if not valid_archives:
        print("ERROR: No valid zip archives provided!")
        return 1

    print(f"Processing {len(valid_archives)} archive(s)...")
    analyze_sequences(valid_archives, args.output, args.target_variance, welch_config)
    return 0


if __name__ == "__main__":
    exit(main())
