"""ML model training, evaluation, and prediction module."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config_loader import config


class ModelTrainer:
    """Handles model training, evaluation, and prediction."""

    def __init__(self):
        self.classes = config.get('classes.names', [])
        self.n_classes = config.get('classes.n_classes', 5)
        self.random_state = config.get('split.random_state', 42)
        self.target_accuracy = config.get('evaluation.target_accuracy', 0.90)
        self.target_f1 = config.get('evaluation.target_f1', 0.79)
        self.cv_folds = config.get('model.cv_folds', 5)

        # Model configurations from config
        self.model_configs = {
            'random_forest': config.get('random_forest', {}),
            'svm': config.get('svm', {}),
            'logistic_regression': config.get('logistic_regression', {}),
            'knn': config.get('knn', {}),
            'gradient_boosting': config.get('gradient_boosting', {}),
        }

        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Results storage
        self.results = {}
        self.cv_results = {}

    def get_model(self, name: str):
        """Get model instance by name with config parameters."""
        params = self.model_configs.get(name, {})

        if name == 'random_forest':
            return RandomForestClassifier(**params)
        elif name == 'svm':
            return SVC(**params)
        elif name == 'logistic_regression':
            return LogisticRegression(**params)
        elif name == 'knn':
            return KNeighborsClassifier(**params)
        elif name == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {name}")

    def create_pipeline(self, model_name: str) -> Pipeline:
        """Create a pipeline with scaler and model."""
        model = self.get_model(model_name)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

    def train_single_model(self, model_name: str, X_train: np.ndarray,
                           y_train: np.ndarray) -> Pipeline:
        """Train a single model."""
        print(f"\nTraining {model_name}...")
        pipeline = self.create_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        self.models[model_name] = pipeline
        return pipeline

    def evaluate_model(self, model_name: str, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained model."""
        pipeline = self.models[model_name]
        y_pred = pipeline.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
        }

        self.results[model_name] = metrics
        return metrics

    def cross_validate_model(self, model_name: str, X: np.ndarray,
                             y: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform cross-validation on a model."""
        pipeline = self.create_pipeline(model_name)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        scoring = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        cv_results = {}

        for metric in scoring:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = scores
            print(f"  {model_name} CV {metric}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        self.cv_results[model_name] = cv_results
        return cv_results

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train and evaluate all configured models."""
        algorithms = config.get('model.algorithms', ['random_forest'])

        print(f"\n{'='*60}")
        print(f"Training {len(algorithms)} models...")
        print(f"{'='*60}")

        for model_name in algorithms:
            # Train
            self.train_single_model(model_name, X_train, y_train)

            # Evaluate on holdout test set
            test_metrics = self.evaluate_model(model_name, X_test, y_test)

            print(f"  Holdout Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Holdout F1 (macro): {test_metrics['f1_macro']:.4f}")

        return self.results

    def cross_validate_all(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Cross-validate all configured models on the full dataset.

        The 50-sample holdout split is too noisy to trust on its own, so
        stratified CV on the full dataset is the headline reliability metric.
        """
        algorithms = config.get('model.algorithms', ['random_forest'])

        print(f"\n{'='*60}")
        print(f"Cross-validating {len(algorithms)} models on full dataset "
              f"({self.cv_folds}-fold stratified)...")
        print(f"{'='*60}")

        for model_name in algorithms:
            self.cross_validate_model(model_name, X, y)

        return self.cv_results

    def select_best_model(self, X_test: np.ndarray = None,
                          y_test: np.ndarray = None) -> str:
        """Select best model.

        Prefers CV scores on the full dataset (unbiased with small data);
        falls back to holdout test metrics when CV hasn't been run.
        """
        if self.cv_results:
            best_score, best_name = -1, None
            for name, cv_res in self.cv_results.items():
                score = cv_res['accuracy'].mean() + cv_res['f1_macro'].mean()
                if score > best_score:
                    best_score = score
                    best_name = name

            self.best_model_name = best_name
            self.best_model = self.models[best_name]

            acc = self.cv_results[best_name]['accuracy']
            f1 = self.cv_results[best_name]['f1_macro']
            print(f"\n{'='*60}")
            print(f"Best model (by {self.cv_folds}-fold CV): {best_name}")
            print(f"  CV Accuracy:   {acc.mean():.4f} (+/- {acc.std():.4f})")
            print(f"  CV F1 (macro): {f1.mean():.4f} (+/- {f1.std():.4f})")
            print(f"{'='*60}")
            return best_name

        if X_test is None or y_test is None:
            raise ValueError("Provide X_test/y_test or run cross-validation first.")

        best_score = -1
        best_name = None

        for name, metrics in self.results.items():
            # Combined score: accuracy + f1_macro
            score = metrics['accuracy'] + metrics['f1_macro']
            if score > best_score:
                best_score = score
                best_name = name

        self.best_model_name = best_name
        self.best_model = self.models[best_name]

        print(f"\n{'='*60}")
        print(f"Best model: {best_name}")
        print(f"  Accuracy: {self.results[best_name]['accuracy']:.4f}")
        print(f"  F1 (macro): {self.results[best_name]['f1_macro']:.4f}")
        print(f"{'='*60}")

        return best_name

    def hyperparameter_tune(self, model_name: str, X_train: np.ndarray,
                            y_train: np.ndarray) -> Pipeline:
        """Perform hyperparameter tuning for a specific model."""
        print(f"\nHyperparameter tuning for {model_name}...")

        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
            },
            'svm': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
                'classifier__kernel': ['rbf', 'poly'],
            },
            'logistic_regression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['lbfgs', 'saga'],
            },
            'knn': {
                'classifier__n_neighbors': [3, 5, 7, 9, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
            },
            'gradient_boosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
            },
        }

        param_grid = param_grids.get(model_name, {})
        if not param_grid:
            print(f"  No parameter grid for {model_name}, skipping tuning")
            return self.models[model_name]

        pipeline = self.create_pipeline(model_name)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")

        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray,
                              save_path: Optional[str] = None):
        """Plot confusion matrix for the best model."""
        if self.best_model is None:
            print("No best model selected yet!")
            return

        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()

    def plot_model_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of all trained models."""
        if not self.results:
            print("No results to plot!")
            return

        metrics_df = pd.DataFrame(self.results).T
        metrics_df = metrics_df[['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot
        metrics_df[['accuracy', 'f1_macro', 'f1_weighted']].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Comparison - Test Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc='lower right')
        axes[0].tick_params(axis='x', rotation=45)

        # CV scores comparison
        if self.cv_results:
            cv_means = {}
            for name, cv_res in self.cv_results.items():
                cv_means[name] = {k: v.mean() for k, v in cv_res.items()}
            cv_df = pd.DataFrame(cv_means).T
            cv_df[['accuracy', 'f1_macro', 'f1_weighted']].plot(kind='bar', ax=axes[1])
            axes[1].set_title('Model Comparison - CV Metrics (Mean)')
            axes[1].set_ylabel('Score')
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(loc='lower right')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        plt.show()

    def plot_feature_importance(self, save_path: Optional[str] = None):
        """Plot feature importance for tree-based models."""
        if self.best_model is None:
            return

        classifier = self.best_model.named_steps['classifier']

        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feature_names = [f'MFCC_{i+1}' for i in range(len(importances))]

            # Get top 20 features
            top_idx = np.argsort(importances)[-20:]
            top_importances = importances[top_idx]
            top_names = [feature_names[i] for i in top_idx]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_importances)), top_importances)
            plt.yticks(range(len(top_importances)), top_names)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {self.best_model_name}')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Feature importance plot saved to {save_path}")
            plt.show()

    def save_model(self, model_path: str):
        """Save the best model to disk."""
        if self.best_model is None:
            print("No model to save!")
            return

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'classes': self.classes,
            'config': config.config,
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load a saved model from disk."""
        model_data = joblib.load(model_path)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.classes = model_data['classes']
        print(f"Model loaded from {model_path}")
        print(f"Model type: {self.best_model_name}")
        print(f"Classes: {self.classes}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict instrument classes for given features."""
        if self.best_model is None:
            raise ValueError("No model loaded! Train or load a model first.")

        predictions = self.best_model.predict(X)
        # Convert numeric labels back to class names
        return np.array([self.classes[p] for p in predictions])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.best_model is None:
            raise ValueError("No model loaded!")

        return self.best_model.predict_proba(X)

    def get_classification_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Get detailed classification report."""
        if self.best_model is None:
            return "No model loaded!"

        y_pred = self.best_model.predict(X_test)
        return classification_report(y_test, y_pred, target_names=self.classes)


def prepare_data_split(X: np.ndarray, y: np.ndarray) -> Tuple:
    """Split data into train/test sets."""
    train_ratio = config.get('split.train_ratio', 0.8)
    test_ratio = config.get('split.test_ratio', 0.2)
    random_state = config.get('split.random_state', 42)
    stratify = config.get('split.stratify', True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_ratio,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y if stratify else None
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Classes: {np.unique(y_train)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the model trainer
    trainer = ModelTrainer()
    print("ModelTrainer initialized")
    print(f"Algorithms: {config.get('model.algorithms')}")