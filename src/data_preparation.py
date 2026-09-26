"""Data preparation script - creates dataset from audio files or generates synthetic data."""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))
from audio_processor import AudioProcessor, create_synthetic_data
from config_loader import config


def prepare_dataset(use_synthetic: bool = False):
    """
    Prepare the dataset by extracting MFCC features from audio files.

    Args:
        use_synthetic: If True, generate synthetic audio data first
    """
    # Get configuration
    classes = config.get('classes.names', [])
    samples_per_class = config.get('classes.samples_per_class', 50)
    raw_audio_dir = config.get('data.raw_audio_dir', 'data/raw_audio')
    features_file = config.get('data.features_file', 'data/processed/features.csv')
    labels_file = config.get('data.labels_file', 'data/processed/labels.csv')

    print(f"{'='*60}")
    print("INSTRUMENT DETECTION - DATA PREPARATION")
    print(f"{'='*60}")
    print(f"Classes: {classes}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Total samples: {len(classes) * samples_per_class}")

    # Create directories
    Path(raw_audio_dir).mkdir(parents=True, exist_ok=True)
    Path(features_file).parent.mkdir(parents=True, exist_ok=True)

    processor = AudioProcessor()

    if use_synthetic:
        print("\n[1/3] Generating synthetic audio data...")
        create_synthetic_data(raw_audio_dir, classes, samples_per_class,
                              config.get('data.sample_rate', 22050),
                              config.get('data.duration', 3.0))
    else:
        print("\n[1/3] Checking for existing audio files...")
        # Check if audio files exist
        total_files = 0
        for class_name in classes:
            class_dir = Path(raw_audio_dir) / class_name
            if class_dir.exists():
                files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
                total_files += len(files)
                print(f"  {class_name}: {len(files)} files")
            else:
                print(f"  {class_name}: directory not found")

        if total_files < len(classes) * samples_per_class * 0.5:
            print(f"\nWarning: Only {total_files} audio files found.")
            print("Consider using --synthetic flag to generate test data.")
            response = input("Generate synthetic data instead? (y/n): ")
            if response.lower() == 'y':
                create_synthetic_data(raw_audio_dir, classes, samples_per_class,
                                      config.get('data.sample_rate', 22050),
                                      config.get('data.duration', 3.0))
            else:
                print("Exiting. Please add audio files to data/raw_audio/<class>/")
                return

    print("\n[2/3] Extracting MFCC features...")
    X, y = processor.process_directory(raw_audio_dir, classes, samples_per_class)

    print("\n[3/3] Saving features...")
    processor.save_features(X, y, features_file, labels_file)

    # Print summary
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Features saved to: {features_file}")
    print(f"Labels saved to: {labels_file}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {classes[cls_idx]}: {count} samples")

    return X, y


def load_prepared_data():
    """Load previously prepared features and labels."""
    features_file = config.get('data.features_file', 'data/processed/features.csv')
    labels_file = config.get('data.labels_file', 'data/processed/labels.csv')

    processor = AudioProcessor()
    X, y = processor.load_features(features_file, labels_file)

    print(f"Loaded features: {X.shape}")
    print(f"Loaded labels: {y.shape}")

    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare instrument detection dataset")
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic audio data for testing')
    parser.add_argument('--load-only', action='store_true',
                        help='Load existing prepared data')

    args = parser.parse_args()

    if args.load_only:
        load_prepared_data()
    else:
        prepare_dataset(use_synthetic=args.synthetic)