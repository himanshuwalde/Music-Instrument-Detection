"""Audio processing and MFCC feature extraction module."""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config_loader import config


class AudioProcessor:
    """Handles audio loading, preprocessing, and MFCC feature extraction."""

    def __init__(self):
        self.sample_rate = config.get('data.sample_rate', 22050)
        self.duration = config.get('data.duration', 3.0)
        self.n_mfcc = config.get('data.n_mfcc', 50)
        self.n_fft = config.get('data.n_fft', 2048)
        self.hop_length = config.get('data.hop_length', 512)
        self.target_samples = int(self.sample_rate * self.duration)

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_signal, sample_rate)
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            return y, sr
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def pad_or_trim(self, y: np.ndarray) -> np.ndarray:
        """Pad or trim audio to target duration."""
        if len(y) > self.target_samples:
            return y[:self.target_samples]
        elif len(y) < self.target_samples:
            return np.pad(y, (0, self.target_samples - len(y)), mode='constant')
        return y

    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features from audio signal.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            MFCC feature vector of shape (n_mfcc,)
        """
        # Pad or trim to consistent length
        y = self.pad_or_trim(y)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Take mean across time axis to get fixed-size feature vector
        mfcc_mean = np.mean(mfccs, axis=1)

        return mfcc_mean

    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """Extract MFCC features from a single audio file."""
        y, sr = self.load_audio(file_path)
        return self.extract_mfcc(y, sr)

    def process_directory(self, data_dir: str, classes: List[str],
                          samples_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all audio files in class subdirectories.

        Expected structure:
        data_dir/
            guitar/
                guitar_001.wav
                guitar_002.wav
                ...
            piano/
                piano_001.wav
                ...

        Args:
            data_dir: Root directory containing class subdirectories
            classes: List of class names (subdirectory names)
            samples_per_class: Number of samples per class

        Returns:
            Tuple of (features_array, labels_array)
        """
        data_dir = Path(data_dir)
        features_list = []
        labels_list = []

        print(f"Processing audio files from {data_dir}...")

        for class_idx, class_name in enumerate(classes):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue

            # Get all audio files
            audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3")) + \
                          list(class_dir.glob("*.flac")) + list(class_dir.glob("*.ogg"))

            # Limit to samples_per_class
            audio_files = audio_files[:samples_per_class]

            print(f"  {class_name}: {len(audio_files)} files")

            for audio_file in tqdm(audio_files, desc=f"  Processing {class_name}"):
                try:
                    features = self.extract_features_from_file(str(audio_file))
                    features_list.append(features)
                    labels_list.append(class_idx)
                except Exception as e:
                    print(f"    Error processing {audio_file.name}: {e}")

        if not features_list:
            raise ValueError("No valid audio files processed!")

        X = np.array(features_list)
        y = np.array(labels_list)

        print(f"\nExtracted features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        return X, y

    def save_features(self, X: np.ndarray, y: np.ndarray,
                      features_path: str, labels_path: str):
        """Save features and labels to CSV files."""
        # Save features
        feature_cols = [f'mfcc_{i+1}' for i in range(X.shape[1])]
        df_features = pd.DataFrame(X, columns=feature_cols)
        df_features.to_csv(features_path, index=False)

        # Save labels
        df_labels = pd.DataFrame(y, columns=['label'])
        df_labels.to_csv(labels_path, index=False)

        print(f"Features saved to {features_path}")
        print(f"Labels saved to {labels_path}")

    def load_features(self, features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels from CSV files."""
        X = pd.read_csv(features_path).values
        y = pd.read_csv(labels_path).values.ravel()
        return X, y


def create_synthetic_data(output_dir: str, classes: List[str],
                          samples_per_class: int, sample_rate: int = 22050,
                          duration: float = 3.0) -> None:
    """
    Create synthetic audio data for testing when real audio files aren't available.
    Generates different waveform types for each instrument class.
    """
    import soundfile as sf
    output_dir = Path(output_dir)

    print("Creating synthetic audio data for testing...")

    for class_idx, class_name in enumerate(classes):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            t = np.linspace(0, duration, int(sample_rate * duration))

            # Generate different synthetic waveforms for each instrument
            if class_name == 'guitar':
                # Guitar-like: decaying harmonics
                signal = np.zeros_like(t)
                for harmonic in [1, 2, 3, 4, 5]:
                    freq = 110 * harmonic  # A2 base
                    decay = np.exp(-t * 2)
                    signal += (1/harmonic) * np.sin(2 * np.pi * freq * t) * decay
                signal += 0.05 * np.random.randn(len(t))  # Noise

            elif class_name == 'piano':
                # Piano-like: sharp attack, harmonic series
                signal = np.zeros_like(t)
                for harmonic in [1, 2, 3, 4, 5, 6]:
                    freq = 220 * harmonic  # A3 base
                    attack = np.exp(-t * 50)  # Sharp attack
                    sustain = np.exp(-t * 1)
                    signal += (1/harmonic) * np.sin(2 * np.pi * freq * t) * (attack + 0.3 * sustain)
                signal += 0.03 * np.random.randn(len(t))

            elif class_name == 'violin':
                # Violin-like: sustained with vibrato
                signal = np.zeros_like(t)
                base_freq = 440  # A4
                vibrato_rate = 5  # Hz
                vibrato_depth = 0.02
                for harmonic in [1, 2, 3, 4]:
                    freq = base_freq * harmonic
                    vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth * freq
                    signal += (1/harmonic) * np.sin(2 * np.pi * (freq + vibrato) * t) * np.exp(-t * 0.5)
                signal += 0.02 * np.random.randn(len(t))

            elif class_name == 'flute':
                # Flute-like: pure sine with breath noise
                signal = np.zeros_like(t)
                base_freq = 440  # A4
                for harmonic in [1, 2, 3]:
                    freq = base_freq * harmonic
                    signal += (1/harmonic) * np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.3)
                # Breath noise
                breath = np.random.randn(len(t)) * 0.1
                breath = np.convolve(breath, np.exp(-np.arange(1000)/100), mode='same')
                signal += breath[:len(t)]

            elif class_name == 'drum':
                # Drum-like: noise burst with pitch drop
                signal = np.random.randn(len(t)) * np.exp(-t * 30)
                # Add a pitch-dropping tone
                pitch = 200 * np.exp(-t * 20)
                signal += 0.5 * np.sin(2 * np.pi * np.cumsum(pitch) / sample_rate) * np.exp(-t * 10)

            # Normalize
            signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8

            # Save
            filename = class_dir / f"{class_name}_{i+1:03d}.wav"
            sf.write(str(filename), signal.astype(np.float32), sample_rate)

        print(f"  Created {samples_per_class} {class_name} samples")

    print(f"\nSynthetic data created in {output_dir}")


if __name__ == "__main__":
    # Test the audio processor
    processor = AudioProcessor()
    print("AudioProcessor initialized")
    print(f"Sample rate: {processor.sample_rate}")
    print(f"Duration: {processor.duration}s")
    print(f"MFCC features: {processor.n_mfcc}")