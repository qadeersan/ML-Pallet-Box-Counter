"""
Dataset Download Script for Pallet Box Counter

This script downloads the datasets from Roboflow Universe.
The datasets are already included in the repository, so this is only needed
if you want to re-download or get the latest version.

Usage:
    python data/download_data.py

Requirements:
    pip install roboflow

Note: You'll need a Roboflow account and API key to download datasets.
"""

import os
from pathlib import Path


def download_datasets():
    """Download datasets from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Please install roboflow: pip install roboflow")
        return

    # Get API key from environment or prompt
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = input("Enter your Roboflow API key: ")

    rf = Roboflow(api_key=api_key)

    # Project root directory
    root_dir = Path(__file__).parent.parent

    print("=" * 60)
    print("Downloading Dataset 1: Boxes")
    print("=" * 60)

    try:
        project1 = rf.workspace("box-fh0kz").project("boxes-ou3c2")
        dataset1 = project1.version(1).download(
            "yolov8",
            location=str(root_dir / "Boxes.v1i.yolov8")
        )
        print(f"✓ Downloaded to: {dataset1.location}")
    except Exception as e:
        print(f"✗ Failed to download Dataset 1: {e}")

    print("\n" + "=" * 60)
    print("Downloading Dataset 2: Final_Object_Detection")
    print("=" * 60)

    try:
        project2 = rf.workspace("vishnua").project("final_object_detection-pjovt")
        dataset2 = project2.version(1).download(
            "yolov8",
            location=str(root_dir / "Final_Object_Detection.v1i.yolov8")
        )
        print(f"✓ Downloaded to: {dataset2.location}")
    except Exception as e:
        print(f"✗ Failed to download Dataset 2: {e}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nDataset Statistics:")
    print("- Boxes.v1i.yolov8: ~5,006 images")
    print("- Final_Object_Detection.v1i.yolov8: ~1,869 images")
    print("- Combined Total: ~6,875 images")


def verify_datasets():
    """Verify that datasets exist and have the expected structure."""
    root_dir = Path(__file__).parent.parent

    datasets = [
        ("Boxes.v1i.yolov8", 5006),
        ("Final_Object_Detection.v1i.yolov8", 1869)
    ]

    print("=" * 60)
    print("Verifying Datasets")
    print("=" * 60)

    all_valid = True
    for dataset_name, expected_count in datasets:
        dataset_path = root_dir / dataset_name

        if not dataset_path.exists():
            print(f"✗ {dataset_name}: NOT FOUND")
            all_valid = False
            continue

        # Count images
        train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
        val_images = list((dataset_path / "valid" / "images").glob("*.jpg"))
        test_images = list((dataset_path / "test" / "images").glob("*.jpg"))
        total = len(train_images) + len(val_images) + len(test_images)

        if total > 0:
            print(f"✓ {dataset_name}: {total} images")
            print(f"  - Train: {len(train_images)}")
            print(f"  - Valid: {len(val_images)}")
            print(f"  - Test: {len(test_images)}")
        else:
            print(f"✗ {dataset_name}: No images found")
            all_valid = False

    return all_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download or verify datasets")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets, don't download"
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_datasets()
    else:
        # First verify if datasets exist
        print("Checking if datasets already exist...\n")
        if verify_datasets():
            print("\n✓ All datasets already present. Use --verify-only to just check.")
            response = input("\nDo you want to re-download anyway? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download.")
                exit(0)

        download_datasets()

