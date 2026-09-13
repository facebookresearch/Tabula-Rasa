# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import Imath
    import OpenEXR
except ImportError:
    print("Error: OpenEXR and Imath packages are required.")
    print("Install with: pip install OpenEXR")
    sys.exit(1)


@dataclass
class FrameError:
    """Stores error metrics for a single frame pair."""

    frame_name: str
    mse: float
    frame_index: int


@dataclass
class OverallStats:
    """Stores aggregate statistics across all frames."""

    average_mse: float
    min_mse: float
    max_mse: float
    std_mse: float
    num_frames: int


def read_exr_image(filepath: str) -> np.ndarray:
    """
    Read an OpenEXR image and return as numpy array.

    Args:
        filepath: Path to the EXR file

    Returns:
        numpy array of shape (height, width, channels) with float32 values
    """
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()

    # Get image dimensions
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Determine available channels
    channels = header["channels"].keys()

    # Read channels (commonly R, G, B, and optionally A)
    channel_data = {}
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    for channel in channels:
        raw_data = exr_file.channel(channel, pt)
        channel_data[channel] = np.frombuffer(raw_data, dtype=np.float32).reshape(
            height, width
        )

    exr_file.close()

    # Stack channels into image array
    # Handle common channel naming conventions
    if "R" in channel_data and "G" in channel_data and "B" in channel_data:
        if "A" in channel_data:
            image = np.stack(
                [
                    channel_data["R"],
                    channel_data["G"],
                    channel_data["B"],
                    channel_data["A"],
                ],
                axis=-1,
            )
        else:
            image = np.stack(
                [channel_data["R"], channel_data["G"], channel_data["B"]], axis=-1
            )
    elif "Y" in channel_data:
        # Grayscale image
        image = channel_data["Y"][..., np.newaxis]
    else:
        # Stack all available channels
        image = np.stack(list(channel_data.values()), axis=-1)

    return image


def get_exr_files_from_directory(directory: str) -> Dict[str, str]:
    """
    Get all EXR files from a directory.

    Args:
        directory: Path to directory

    Returns:
        Dictionary mapping filename (without extension) to full path
    """
    exr_files = {}
    dir_path = Path(directory)

    for filepath in dir_path.glob("*.exr"):
        name = filepath.stem
        exr_files[name] = str(filepath)

    # Also check for uppercase extension
    for filepath in dir_path.glob("*.EXR"):
        name = filepath.stem
        if name not in exr_files:
            exr_files[name] = str(filepath)

    return exr_files


def get_exr_files_from_zip(zip_path: str, temp_dir: str) -> Dict[str, str]:
    """
    Extract EXR files from a zip archive to a temporary directory.

    Args:
        zip_path: Path to zip archive
        temp_dir: Temporary directory to extract files to

    Returns:
        Dictionary mapping filename (without extension) to extracted file path
    """
    exr_files = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".exr"):
                # Extract to temp directory
                extracted_path = zf.extract(name, temp_dir)
                # Get just the filename without path and extension
                base_name = Path(name).stem
                exr_files[base_name] = extracted_path

    return exr_files


def get_exr_files(source: str, temp_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get EXR files from either a directory or zip archive.

    Args:
        source: Path to directory or zip file
        temp_dir: Temporary directory for zip extraction (required for zip files)

    Returns:
        Dictionary mapping filename (without extension) to full path
    """
    if zipfile.is_zipfile(source):
        if temp_dir is None:
            raise ValueError("temp_dir is required for zip file extraction")
        return get_exr_files_from_zip(source, temp_dir)
    elif os.path.isdir(source):
        return get_exr_files_from_directory(source)
    else:
        raise ValueError(f"Source '{source}' is neither a directory nor a zip file")


def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Mean Square Error between two images.

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array

    Returns:
        MSE value as float
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes don't match: {image1.shape} vs {image2.shape}")

    print(f"{np.count_nonzero(np.isnan(image1))}_{np.count_nonzero(np.isnan(image2))}")
    diff = image1.astype(np.float64) - image2.astype(np.float64)
    mse = np.mean(diff**2)
    return float(mse)


def calculate_signed_difference(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Calculate signed difference between two images.

    Args:
        image1: First image (reference)
        image2: Second image (comparison)

    Returns:
        Signed difference array (image2 - image1)
    """
    return image2.astype(np.float64) - image1.astype(np.float64)


def apply_false_color(diff_image: np.ndarray, max_scale: float) -> np.ndarray:
    """
    Apply false color mapping to signed difference image.

    Uses matplotlib's "coolwarm" diverging colormap where:
    - Negative differences -> Blue
    - Zero -> White
    - Positive differences -> Red

    Args:
        diff_image: Signed difference image (can contain negative values)
        max_scale: Maximum absolute value for normalization. Values are
            clamped to [-max_scale, max_scale] before mapping.

    Returns:
        RGB image with false color mapping (uint8)
    """
    try:
        import matplotlib
    except ImportError:
        print("Warning: matplotlib not available, cannot generate false color images")
        print("Install with: pip install matplotlib")
        return None

    # Average across channels if multi-channel
    if diff_image.ndim == 3:
        diff_scalar = np.mean(diff_image, axis=-1)
    else:
        diff_scalar = diff_image

    # Normalize to [0, 1] range for colormap
    # Map [-max_scale, max_scale] -> [0, 1]
    if max_scale <= 0:
        max_scale = 1.0
    normalized = np.clip(diff_scalar / max_scale, -1, 1)
    # Convert from [-1, 1] to [0, 1] for colormap lookup
    normalized_01 = (normalized + 1) / 2

    # Get the coolwarm colormap from matplotlib
    cmap = matplotlib.colormaps["bwr"]

    # Apply colormap (returns RGBA float values in [0, 1])
    false_color_rgba = cmap(normalized_01)

    # Convert to RGB uint8 (drop alpha channel)
    false_color_uint8 = (false_color_rgba[..., :3] * 255).astype(np.uint8)

    return false_color_uint8


def save_false_color_image(image: np.ndarray, filepath: str) -> None:
    """
    Save false color image to file.

    Args:
        image: RGB image as numpy array (uint8)
        filepath: Output file path (PNG format)
    """
    try:
        from PIL import Image

        img = Image.fromarray(image, mode="RGB")
        img.save(filepath)
    except ImportError:
        print("Warning: PIL not available, cannot save false color images")
        print("Install with: pip install Pillow")


def write_csv_output(
    frame_errors: List[FrameError],
    overall_stats: OverallStats,
    output_path: str,
) -> None:
    """
    Write error values to CSV file.

    Args:
        frame_errors: List of per-frame error data
        overall_stats: Aggregate statistics
        output_path: Path to output CSV file
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Frame Index", "Frame Name", "MSE"])

        # Write per-frame data (sorted by frame index)
        sorted_errors = sorted(frame_errors, key=lambda x: x.frame_index)
        for error in sorted_errors:
            writer.writerow([error.frame_index, error.frame_name, error.mse])

        # Write separator and overall statistics
        writer.writerow([])
        writer.writerow(["Overall Statistics"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Number of Frames", overall_stats.num_frames])
        writer.writerow(["Average MSE", overall_stats.average_mse])
        writer.writerow(["Min MSE", overall_stats.min_mse])
        writer.writerow(["Max MSE", overall_stats.max_mse])
        writer.writerow(["Std MSE", overall_stats.std_mse])


def plot_mse_errors(
    frame_errors: List[FrameError],
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Plot MSE errors vs frame index.

    Args:
        frame_errors: List of per-frame error data
        output_path: Optional path to save plot image
        show_plot: Whether to display the plot interactively
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, cannot generate plot")
        print("Install with: pip install matplotlib")
        return

    # Sort by frame index
    sorted_errors = sorted(frame_errors, key=lambda x: x.frame_index)
    indices = [e.frame_index for e in sorted_errors]
    mse_values = [e.mse for e in sorted_errors]

    plt.figure(figsize=(12, 6))
    plt.plot(indices, mse_values, "b-o", markersize=4, linewidth=1)
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Square Error (MSE)")
    plt.title("MSE Error vs Frame Index")
    plt.grid(True, alpha=0.3)

    # Add statistics annotations
    avg_mse = np.mean(mse_values)
    plt.axhline(y=avg_mse, color="r", linestyle="--", label=f"Average: {avg_mse:.6f}")
    plt.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def extract_frame_index(filename: str) -> int:
    """
    Extract frame index from filename.

    Attempts to find a number in the filename that represents the frame index.

    Args:
        filename: Filename without extension

    Returns:
        Extracted frame index, or hash-based index if no number found
    """
    import re

    # Try to find numbers in the filename
    numbers = re.findall(r"\d+", filename)
    if numbers:
        # Use the last number as frame index (common convention)
        return int(numbers[-1])
    else:
        # Fall back to hash-based ordering
        return hash(filename) % 10000


def compare_sequences(
    source1: str,
    source2: str,
    output_csv: str,
    diff_output_dir: Optional[str] = None,
    diff_scale: float = 1.0,
    plot_output: Optional[str] = None,
    show_plot: bool = False,
) -> Tuple[List[FrameError], OverallStats]:
    """
    Compare two image sequences and calculate MSE errors.

    Args:
        source1: Path to first sequence (directory or zip)
        source2: Path to second sequence (directory or zip)
        output_csv: Path to output CSV file
        diff_output_dir: Optional directory for difference images
        diff_scale: Maximum absolute scale for false color mapping
        plot_output: Optional path to save plot image
        show_plot: Whether to display plot interactively

    Returns:
        Tuple of (frame_errors, overall_stats)
    """
    # Create temporary directories for zip extraction if needed
    temp_dirs = []

    try:
        # Get EXR files from both sources
        temp_dir1 = None
        temp_dir2 = None

        if zipfile.is_zipfile(source1):
            temp_dir1 = tempfile.mkdtemp(prefix="seq1_")
            temp_dirs.append(temp_dir1)

        if zipfile.is_zipfile(source2):
            temp_dir2 = tempfile.mkdtemp(prefix="seq2_")
            temp_dirs.append(temp_dir2)

        print(f"Loading sequence 1 from: {source1}")
        files1 = get_exr_files(source1, temp_dir1)
        print(f"  Found {len(files1)} EXR files")

        print(f"Loading sequence 2 from: {source2}")
        files2 = get_exr_files(source2, temp_dir2)
        print(f"  Found {len(files2)} EXR files")

        # Find matching frames
        names1 = set(files1.keys())
        names2 = set(files2.keys())

        matched_names = names1 & names2
        unmatched_in_1 = names1 - names2
        unmatched_in_2 = names2 - names1

        # Warn about unmatched frames
        if unmatched_in_1:
            print(
                f"\nWarning: {len(unmatched_in_1)} frames in sequence 1 have no match:"
            )
            for name in sorted(unmatched_in_1)[:10]:
                print(f"  - {name}")
            if len(unmatched_in_1) > 10:
                print(f"  ... and {len(unmatched_in_1) - 10} more")

        if unmatched_in_2:
            print(
                f"\nWarning: {len(unmatched_in_2)} frames in sequence 2 have no match:"
            )
            for name in sorted(unmatched_in_2)[:10]:
                print(f"  - {name}")
            if len(unmatched_in_2) > 10:
                print(f"  ... and {len(unmatched_in_2) - 10} more")

        if len(files1) != len(files2):
            print(
                f"\nWarning: Sequences have different frame counts: "
                f"{len(files1)} vs {len(files2)}"
            )

        if not matched_names:
            print("Error: No matching frames found between sequences!")
            sys.exit(1)

        print(f"\nProcessing {len(matched_names)} matching frame pairs...")

        # Create diff output directory if specified
        if diff_output_dir:
            os.makedirs(diff_output_dir, exist_ok=True)
            print(f"Difference images will be saved to: {diff_output_dir}")

        # Process each matching pair
        frame_errors = []
        for i, name in enumerate(sorted(matched_names)):
            path1 = files1[name]
            path2 = files2[name]

            # Read images
            image1 = read_exr_image(path1)
            image2 = read_exr_image(path2)

            # Calculate MSE
            mse = calculate_mse(image1, image2)

            # Extract frame index from filename
            frame_idx = extract_frame_index(name)

            frame_errors.append(
                FrameError(frame_name=name, mse=mse, frame_index=frame_idx)
            )

            # Generate difference image if requested
            if diff_output_dir:
                diff = calculate_signed_difference(image1, image2)
                false_color = apply_false_color(diff, diff_scale)
                if false_color is not None:
                    diff_path = os.path.join(diff_output_dir, f"{name}_diff.png")
                    save_false_color_image(false_color, diff_path)

            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(matched_names):
                print(f"  Processed {i + 1}/{len(matched_names)} frames", end="\r")

        print()  # New line after progress

        # Calculate overall statistics
        mse_values = [e.mse for e in frame_errors]
        overall_stats = OverallStats(
            average_mse=float(np.mean(mse_values)),
            min_mse=float(np.min(mse_values)),
            max_mse=float(np.max(mse_values)),
            std_mse=float(np.std(mse_values)),
            num_frames=len(frame_errors),
        )

        # Write CSV output
        write_csv_output(frame_errors, overall_stats, output_csv)
        print(f"\nResults written to: {output_csv}")

        # Print summary
        print("\n=== Summary ===")
        print(f"Frames compared: {overall_stats.num_frames}")
        print(f"Average MSE: {overall_stats.average_mse:.6e}")
        print(f"Min MSE: {overall_stats.min_mse:.6e}")
        print(f"Max MSE: {overall_stats.max_mse:.6e}")
        print(f"Std MSE: {overall_stats.std_mse:.6e}")

        # Generate plot if requested
        if plot_output or show_plot:
            plot_mse_errors(frame_errors, plot_output, show_plot)

        return frame_errors, overall_stats

    finally:
        # Clean up temporary directories
        import shutil

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two sequences of OpenEXR images and calculate MSE errors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two directories
  %(prog)s dir1/ dir2/ -o results.csv

  # Compare directory with zip archive
  %(prog)s dir1/ archive.zip -o results.csv

  # Generate difference images with custom scale
  %(prog)s dir1/ dir2/ -o results.csv --diff-output diff_images/ --diff-scale 0.5

  # Plot MSE errors
  %(prog)s dir1/ dir2/ -o results.csv --plot --plot-output mse_plot.png
        """,
    )

    parser.add_argument(
        "sequence1",
        help="Path to first image sequence (directory or zip archive)",
    )
    parser.add_argument(
        "sequence2",
        help="Path to second image sequence (directory or zip archive)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--diff-output",
        metavar="DIR",
        help="Directory to save false-color difference images",
    )
    parser.add_argument(
        "--diff-scale",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Maximum absolute scale for false color mapping (default: 1.0). "
        "Values are clamped to [-SCALE, SCALE] before applying colormap.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show MSE plot interactively",
    )
    parser.add_argument(
        "--plot-output",
        metavar="FILE",
        help="Path to save MSE plot image (e.g., plot.png)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.sequence1):
        print(f"Error: Sequence 1 not found: {args.sequence1}")
        sys.exit(1)

    if not os.path.exists(args.sequence2):
        print(f"Error: Sequence 2 not found: {args.sequence2}")
        sys.exit(1)

    # Run comparison
    compare_sequences(
        source1=args.sequence1,
        source2=args.sequence2,
        output_csv=args.output,
        diff_output_dir=args.diff_output,
        diff_scale=args.diff_scale,
        plot_output=args.plot_output,
        show_plot=args.plot,
    )


if __name__ == "__main__":
    main()
