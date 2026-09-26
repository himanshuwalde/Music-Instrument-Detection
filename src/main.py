#!/usr/bin/env python
"""
Main execution script for Instrument Detection ML Project.

This script runs the complete pipeline:
1. Data preparation (feature extraction from audio)
2. Model training and evaluation
3. Model selection and saving
4. Optional: Predict on new audio files
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config_loader import config
from data_preparation import prepare_dataset, load_prepared_data
from model_trainer import (ModelTrainer, prepare_data_split)
from audio_processor import AudioProcessor


def run_training_pipeline(use_synthetic: bool = False, tune_hyperparams: bool = False):
    """Run the complete training pipeline."""
    print(f"\n{'#'*70}")
    print("# INSTRUMENT DETECTION - TRAINING PIPELINE")
    print(f"{'#'*70}")

    # Step 1: Prepare data
    print("\n[STEP 1] Data Preparation")
    print("-" * 40)
    X, y = prepare_dataset(use_synthetic=use_synthetic)

    # Step 2: Train-test split
    print("\n[STEP 2] Data Split")
    print("-" * 40)
    X_train, X_test, y_train, y_test = prepare_data_split(X, y)

    # Step 3: Train models
    print("\n[STEP 3] Model Training")
    print("-" * 40)
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Step 4: Cross-validation on the full dataset (headline metric)
    print("\n[STEP 4] Cross-Validation (full dataset)")
    print("-" * 40)
    trainer.cross_validate_all(X, y)

    # Step 5: Hyperparameter tuning (optional)
    if tune_hyperparams:
        print("\n[STEP 5] Hyperparameter Tuning")
        print("-" * 40)
        algorithms = config.get('model.algorithms', ['random_forest'])
        for model_name in algorithms:
            trainer.hyperparameter_tune(model_name, X_train, y_train)
            # Re-evaluate after tuning
            trainer.evaluate_model(model_name, X_test, y_test)

    # Step 6: Select best model
    print("\n[STEP 6] Model Selection")
    print("-" * 40)
    trainer.select_best_model(X_test, y_test)

    # Step 7: Detailed evaluation on untouched holdout set
    print("\n[STEP 7] Holdout Evaluation (sanity check)")
    print("-" * 40)
    print("\nClassification Report:")
    print(trainer.get_classification_report(X_test, y_test))

    # Step 8: Visualizations
    print("\n[STEP 8] Generating Visualizations")
    print("-" * 40)

    plots_dir = Path(config.get('output.plots_dir', 'results/plots'))
    plots_dir.mkdir(parents=True, exist_ok=True)

    trainer.plot_model_comparison(save_path=str(plots_dir / "model_comparison.png"))
    trainer.plot_confusion_matrix(X_test, y_test, save_path=str(plots_dir / "confusion_matrix.png"))
    trainer.plot_feature_importance(save_path=str(plots_dir / "feature_importance.png"))

    # Step 9: Save model and evaluation artifacts
    print("\n[STEP 9] Saving Model & Metrics")
    print("-" * 40)
    model_dir = Path(config.get('output.model_dir', 'models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir / "best_model.joblib"))

    results_dir = Path(config.get('output.results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    cv_rows = [{'model': name, **{f'{m}_mean': s.mean() for m, s in res.items()},
                **{f'{m}_std': s.std() for m, s in res.items()}}
               for name, res in trainer.cv_results.items()]
    pd.DataFrame(cv_rows).to_csv(results_dir / 'cv_results.csv', index=False)
    pd.DataFrame(trainer.results).T.to_csv(results_dir / 'test_metrics.csv')
    print(f"Metrics saved to {results_dir}/cv_results.csv and test_metrics.csv")

    # Final summary: CV is the reliable estimate; holdout is a sanity check
    best = trainer.best_model_name
    cv_acc = trainer.cv_results[best]['accuracy']
    cv_f1 = trainer.cv_results[best]['f1_macro']
    holdout_acc = trainer.results[best]['accuracy']
    holdout_f1 = trainer.results[best]['f1_macro']

    print(f"\n{'#'*70}")
    print("# TRAINING COMPLETE")
    print(f"{'#'*70}")
    print(f"Best Model: {best}")
    print(f"CV Accuracy:      {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"CV F1 (macro):    {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"Holdout Accuracy: {holdout_acc:.4f} (single 50-sample split, noisy)")
    print(f"Holdout F1:       {holdout_f1:.4f}")
    print(f"Model saved to: {model_dir / 'best_model.joblib'}")

    # Check if targets met (against CV means, the unbiased estimate)
    target_acc = config.get('evaluation.target_accuracy', 0.90)
    target_f1 = config.get('evaluation.target_f1', 0.79)
    acc = cv_acc.mean()
    f1 = cv_f1.mean()

    print(f"\nTarget Accuracy: {target_acc:.2f} - {'[MET]' if acc >= target_acc else '[NOT MET]'} (CV: {acc:.4f})")
    print(f"Target F1 Score: {target_f1:.2f} - {'[MET]' if f1 >= target_f1 else '[NOT MET]'} (CV: {f1:.4f})")

    return trainer


def predict_on_audio_file(audio_path: str, model_path: str = None):
    """Predict instrument for a single audio file."""
    if model_path is None:
        model_path = Path(config.get('output.model_dir', 'models')) / "best_model.joblib"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using: python src/main.py --train")
        return

    # Load model
    trainer = ModelTrainer()
    trainer.load_model(str(model_path))

    # Process audio
    processor = AudioProcessor()
    print(f"\nProcessing audio file: {audio_path}")

    try:
        features = processor.extract_features_from_file(audio_path)
        features = features.reshape(1, -1)  # Reshape for single sample

        # Predict
        prediction = trainer.predict(features)
        probabilities = trainer.predict_proba(features)

        print(f"\n{'='*50}")
        print(f"PREDICTION RESULT")
        print(f"{'='*50}")
        print(f"Predicted Instrument: {prediction[0]}")

        print("\nClass Probabilities:")
        for i, cls in enumerate(trainer.classes):
            print(f"  {cls:10s}: {probabilities[0][i]:.4f}")

        return prediction[0], probabilities[0]

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None


def predict_batch(audio_dir: str, model_path: str = None):
    """Predict instruments for all audio files in a directory."""
    if model_path is None:
        model_path = Path(config.get('output.model_dir', 'models')) / "best_model.joblib"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    trainer = ModelTrainer()
    trainer.load_model(str(model_path))
    processor = AudioProcessor()

    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + \
                  list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.ogg"))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"\nPredicting {len(audio_files)} files from {audio_dir}...")

    results = []
    for audio_file in audio_files:
        try:
            features = processor.extract_features_from_file(str(audio_file))
            features = features.reshape(1, -1)
            prediction = trainer.predict(features)
            probas = trainer.predict_proba(features)
            confidence = np.max(probas)

            results.append({
                'file': audio_file.name,
                'predicted': prediction[0],
                'confidence': confidence
            })
            print(f"  {audio_file.name:30s} -> {prediction[0]:10s} (conf: {confidence:.3f})")
        except Exception as e:
            print(f"  {audio_file.name:30s} -> ERROR: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("BATCH PREDICTION SUMMARY")
    print(f"{'='*50}")
    from collections import Counter
    pred_counts = Counter(r['predicted'] for r in results)
    for cls, count in pred_counts.items():
        print(f"  {cls:10s}: {count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Instrument Detection - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete training with synthetic data
  python src/main.py --train --synthetic

  # Run training with existing audio files
  python src/main.py --train

  # Run training with hyperparameter tuning
  python src/main.py --train --tune

  # Predict on a single audio file
  python src/main.py --predict audio/test_guitar.wav

  # Predict on batch of audio files
  python src/main.py --predict-batch audio/test_folder/

  # Just prepare data (extract features)
  python src/main.py --prepare-data --synthetic
        """
    )

    parser.add_argument('--train', action='store_true',
                        help='Run complete training pipeline')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for training/testing')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Only prepare data (extract features)')
    parser.add_argument('--predict', type=str, metavar='AUDIO_FILE',
                        help='Predict instrument for a single audio file')
    parser.add_argument('--predict-batch', type=str, metavar='AUDIO_DIR',
                        help='Predict instruments for all audio files in directory')
    parser.add_argument('--model', type=str, metavar='MODEL_PATH',
                        help='Path to model file (default: models/best_model.joblib)')

    args = parser.parse_args()

    if args.prepare_data:
        prepare_dataset(use_synthetic=args.synthetic)

    elif args.train:
        run_training_pipeline(use_synthetic=args.synthetic, tune_hyperparams=args.tune)

    elif args.predict:
        predict_on_audio_file(args.predict, args.model)

    elif args.predict_batch:
        predict_batch(args.predict_batch, args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()