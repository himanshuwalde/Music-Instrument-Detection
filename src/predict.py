#!/usr/bin/env python
"""
Standalone prediction script for instrument detection.
Use this to predict instruments on new audio files after training.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from config_loader import config
from model_trainer import ModelTrainer
from audio_processor import AudioProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Predict musical instrument from audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single file
  python src/predict.py audio/guitar_test.wav

  # Predict with custom model
  python src/predict.py audio/test.wav --model models/my_model.joblib

  # Predict batch with output to CSV
  python src/predict.py audio/test_folder/ --batch --output predictions.csv

  # Show top-3 predictions with probabilities
  python src/predict.py audio/test.wav --top-k 3
        """
    )

    parser.add_argument('input', type=str, help='Audio file or directory path')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: models/best_model.joblib)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all audio files in input directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Save predictions to CSV file')
    parser.add_argument('--top-k', type=int, default=1,
                        help='Show top-K predictions with probabilities')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Minimum confidence threshold for predictions')

    args = parser.parse_args()

    # Default model path
    if args.model is None:
        args.model = str(Path(config.get('output.model_dir', 'models')) / "best_model.joblib")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train a model first: python src/main.py --train")
        sys.exit(1)

    # Load model
    print(f"Loading model from {model_path}...")
    trainer = ModelTrainer()
    trainer.load_model(str(model_path))

    processor = AudioProcessor()
    input_path = Path(args.input)

    results = []

    if args.batch:
        # Process directory
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)

        audio_files = list(input_path.glob("*.wav")) + \
                      list(input_path.glob("*.mp3")) + \
                      list(input_path.glob("*.flac")) + \
                      list(input_path.glob("*.ogg"))

        if not audio_files:
            print(f"No audio files found in {input_path}")
            sys.exit(0)

        print(f"Processing {len(audio_files)} audio files...")

        for audio_file in audio_files:
            try:
                features = processor.extract_features_from_file(str(audio_file))
                features = features.reshape(1, -1)

                prediction = trainer.predict(features)[0]
                probabilities = trainer.predict_proba(features)[0]

                # Get top-k predictions
                top_k_idx = np.argsort(probabilities)[-args.top_k:][::-1]
                top_k_classes = [trainer.classes[i] for i in top_k_idx]
                top_k_probs = probabilities[top_k_idx]

                # Filter by threshold
                filtered = [(c, p) for c, p in zip(top_k_classes, top_k_probs) if p >= args.threshold]

                result = {
                    'file': audio_file.name,
                    'predicted': filtered[0][0] if filtered else 'unknown',
                    'confidence': filtered[0][1] if filtered else 0.0
                }

                # Add top-k as separate columns
                for i, (cls, prob) in enumerate(filtered):
                    result[f'top_{i+1}_class'] = cls
                    result[f'top_{i+1}_prob'] = prob

                results.append(result)

                # Print progress
                top_str = ", ".join([f"{c}({p:.3f})" for c, p in filtered[:3]])
                print(f"  {audio_file.name:30s} -> {top_str}")

            except Exception as e:
                print(f"  {audio_file.name:30s} -> ERROR: {e}")
                results.append({'file': audio_file.name, 'predicted': 'error', 'confidence': 0.0})

    else:
        # Process single file
        if not input_path.is_file():
            print(f"Error: {input_path} is not a file")
            sys.exit(1)

        try:
            features = processor.extract_features_from_file(str(input_path))
            features = features.reshape(1, -1)

            prediction = trainer.predict(features)[0]
            probabilities = trainer.predict_proba(features)[0]

            # Get top-k
            top_k_idx = np.argsort(probabilities)[-args.top_k:][::-1]
            top_k_classes = [trainer.classes[i] for i in top_k_idx]
            top_k_probs = probabilities[top_k_idx]

            print(f"\n{'='*50}")
            print(f"PREDICTION RESULT: {input_path.name}")
            print(f"{'='*50}")
            print(f"Predicted Instrument: {prediction}")

            print(f"\nTop-{args.top_k} Predictions:")
            for i, (cls, prob) in enumerate(zip(top_k_classes, top_k_probs)):
                marker = ">" if i == 0 else " "
                print(f"  {marker} {i+1}. {cls:10s}: {prob:.4f} ({prob*100:.1f}%)")

            results.append({
                'file': input_path.name,
                'predicted': prediction,
                'confidence': top_k_probs[0]
            })
            for i, (cls, prob) in enumerate(zip(top_k_classes, top_k_probs)):
                results[0][f'top_{i+1}_class'] = cls
                results[0][f'top_{i+1}_prob'] = prob

        except Exception as e:
            print(f"Error processing audio: {e}")
            sys.exit(1)

    # Save to CSV if requested
    if args.output and results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

    # Summary for batch
    if args.batch and results:
        from collections import Counter
        pred_counts = Counter(r['predicted'] for r in results if r['predicted'] != 'error')
        print(f"\n{'='*50}")
        print("PREDICTION SUMMARY")
        print(f"{'='*50}")
        for cls, count in pred_counts.most_common():
            print(f"  {cls:10s}: {count} files")
        print(f"  Total: {len(results)} files")


if __name__ == "__main__":
    import pandas as pd
    main()