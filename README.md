# 🎵 Instrument Detection from Audio Clips

A Machine Learning project for classifying musical instruments from audio recordings using MFCC (Mel-Frequency Cepstral Coefficients) features.

## 📋 Project Overview

| Aspect | Details |
|--------|---------|
| **Problem Type** | Supervised Multiclass Classification |
| **Classes** | Guitar, Piano, Violin, Flute, Drum (5 instruments) |
| **Features** | 50 MFCC coefficients per audio clip |
| **Dataset** | 250 samples (50 per instrument) |
| **Split** | 80% Train (200) / 20% Test (50) |
| **Target Accuracy** | 90% |
| **Target F1 Score** | 0.79 |

## 🏗️ Project Structure

```
Instrument Detection/
├── config.yaml              # Project configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── src/
│   ├── config_loader.py    # Configuration management
│   ├── audio_processor.py  # Audio loading & MFCC extraction
│   ├── data_preparation.py # Dataset creation from audio files
│   ├── model_trainer.py    # ML model training & evaluation
│   ├── main.py             # Main pipeline entry point
│   └── predict.py          # Standalone prediction script
├── notebooks/
│   ├── 01_explore_data.ipynb    # Data exploration & visualization
│   └── 02_model_training.ipynb  # Model training & evaluation
├── data/
│   ├── raw_audio/          # Place audio files here (organized by class)
│   │   ├── guitar/
│   │   ├── piano/
│   │   ├── violin/
│   │   ├── flute/
│   │   └── drum/
│   └── processed/          # Extracted features (auto-generated)
├── models/                 # Trained models (auto-generated)
└── results/
    └── plots/              # Visualization outputs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

**Option A: Use your own audio files**
Organize your WAV files in `data/raw_audio/`:
```
data/raw_audio/
├── guitar/
│   ├── guitar_001.wav
│   ├── guitar_002.wav
│   └── ...
├── piano/
├── violin/
├── flute/
└── drum/
```

Then run:
```bash
python src/main.py --prepare-data
```

**Option B: Generate synthetic data for testing**
```bash
python src/main.py --prepare-data --synthetic
```

### 3. Train Models

```bash
# Basic training
python src/main.py --train

# With synthetic data
python src/main.py --train --synthetic

# With hyperparameter tuning
python src/main.py --train --tune
```

### 4. Make Predictions

```bash
# Single file
python src/predict.py audio/test_guitar.wav

# Batch prediction
python src/predict.py audio/test_folder/ --batch --output predictions.csv

# Top-3 predictions with probabilities
python src/predict.py audio/test.wav --top-k 3
```

## 📊 Algorithms Supported

| Algorithm | Description |
|-----------|-------------|
| **Random Forest** | Ensemble of decision trees (default) |
| **SVM** | Support Vector Machine with RBF kernel |
| **Logistic Regression** | Multinomial logistic regression |
| **K-Nearest Neighbors** | Distance-based classification |
| **Gradient Boosting** | Boosted decision trees |

## 🔧 Configuration

Modify `config.yaml` to customize:
- Audio processing parameters (sample rate, duration, MFCC count)
- Model hyperparameters
- Train/test split ratios
- Target metrics

## 📈 Expected Results

With the default configuration and synthetic data:
- **Random Forest**: ~85-95% accuracy
- **SVM**: ~80-90% accuracy
- **Logistic Regression**: ~75-85% accuracy

Real audio data typically achieves higher accuracy with proper preprocessing.

## 📓 Jupyter Notebooks

Explore the data and models interactively:

```bash
jupyter notebook notebooks/
```

- **01_explore_data.ipynb**: MFCC visualization, PCA, correlations
- **02_model_training.ipynb**: Model comparison, hyperparameter tuning

## 🎯 Pipeline Flow

```
Audio Clip (.wav)
       │
       ▼
Load & Resample (22.05 kHz, 3s)
       │
       ▼
Extract 50 MFCC Features (Librosa)
       │
       ▼
Mean across time → 50-dim vector
       │
       ▼
StandardScaler Normalization
       │
       ▼
ML Classifier (RF/SVM/LR/KNN/GB)
       │
       ▼
Instrument Prediction 🎸🎹🎻🪈🥁
```

## 📝 Usage Examples

### Using as a Module

```python
from src.audio_processor import AudioProcessor
from src.model_trainer import ModelTrainer

# Load trained model
trainer = ModelTrainer()
trainer.load_model('models/best_model.joblib')

# Predict on new audio
processor = AudioProcessor()
features = processor.extract_features_from_file('my_audio.wav')
features = features.reshape(1, -1)

prediction = trainer.predict(features)
probabilities = trainer.predict_proba(features)

print(f"Instrument: {prediction[0]}")
print(f"Confidence: {probabilities[0].max():.2%}")
```

### Custom Model Training

```python
from src.model_trainer import ModelTrainer
from src.data_preparation import load_prepared_data
from sklearn.model_selection import train_test_split

# Load data
X, y = load_prepared_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train custom model
trainer = ModelTrainer()
trainer.train_single_model('random_forest', X_train, y_train)
metrics = trainer.evaluate_model('random_forest', X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## 🔬 Feature Engineering Details

**MFCC Extraction Parameters:**
- Sample Rate: 22,050 Hz
- Duration: 3 seconds
- N_MFCC: 50 coefficients
- N_FFT: 2048
- Hop Length: 512

**Preprocessing:**
1. Load audio → resample to 22.05 kHz
2. Pad/trim to 3 seconds (66,150 samples)
3. Compute MFCCs using Librosa
4. Average across time frames → 50 features
5. StandardScaler normalization

## 📊 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **F1 Macro**: Unweighted mean of per-class F1
- **F1 Weighted**: Support-weighted mean F1
- **Precision/Recall**: Per-class and macro averages

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: librosa` | Run `pip install -r requirements.txt` |
| No audio files found | Check `data/raw_audio/<class>/` structure |
| Low accuracy | Increase training data, tune hyperparameters, try different features |
| Memory error | Reduce batch size, use `n_jobs=1` in config |

## 📄 License

This project is for educational/practical purposes.

## 🤝 Contributing

1. Add more instrument classes
2. Experiment with additional features (spectral centroid, chroma, etc.)
3. Try deep learning approaches (CNN on spectrograms)
4. Add data augmentation (pitch shift, time stretch, noise)

---

**Happy Classifying!** 🎵