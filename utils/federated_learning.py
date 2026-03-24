# utils/federated_learning.py (FIXED VERSION)
# -*- coding: utf-8 -*-
"""
Federated learning implementation for disability employment matching.

"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
import logging

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    import lightgbm as lgb
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Missing required packages. Please install: pip install scikit-learn lightgbm imbalanced-learn")
    logger.error(f"Import error: {e}")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedModel:
    """
    Wrapper for individual regional models in federated learning setup.
    """
    
    def __init__(self, region_name: str, model: BaseEstimator, 
                 training_samples: int, model_params: Dict = None,
                 preprocessing_objects: Dict = None):
        self.region_name = region_name
        self.model = model
        self.training_samples = training_samples
        self.model_params = model_params or {}
        self.preprocessing_objects = preprocessing_objects or {}
        self.training_time = 0.0
        self.metrics = {}
        
    def predict(self, X):
        """Predict with preprocessing applied."""
        X_processed = self._apply_preprocessing(X)
        return self.model.predict(X_processed)
        
    def predict_proba(self, X):
        """Predict probabilities with preprocessing applied."""
        X_processed = self._apply_preprocessing(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        else:
            # Fallback for models without predict_proba
            preds = self.model.predict(X_processed)
            proba = np.zeros((len(preds), 2))
            proba[:, 1] = preds
            proba[:, 0] = 1 - preds
            return proba
    
    def _apply_preprocessing(self, X):
        """Apply the same preprocessing pipeline as used during training."""
        if not self.preprocessing_objects:
            return X
            
        X_processed = X.copy()
        
        # Apply scaler if available
        if 'scaler' in self.preprocessing_objects:
            X_processed = self.preprocessing_objects['scaler'].transform(X_processed)
            
        # Apply feature selector if available
        if 'selector' in self.preprocessing_objects:
            X_processed = self.preprocessing_objects['selector'].transform(X_processed)
            
        return X_processed


class FederatedTrainer:
    """
    Federated learning trainer for employment matching models.

    - RobustScaler normalization
    - SelectKBest feature selection
    - SMOTE oversampling
    - CalibratedClassifierCV probability calibration
    """
    
    def __init__(self, base_model_class=lgb.LGBMClassifier, **model_params):
        self.base_model_class = base_model_class
        self.model_params = model_params
        self.regional_models = {}
        self.global_model = None
        self.training_history = []
        
        logger.info(f"Initialized FederatedTrainer with {base_model_class.__name__}")

    def train_regional_models(self, regional_datasets: Dict[str, pd.DataFrame],
                            target_column: str = 'outcome',
                            validation_split: float = 0.2,
                            random_state: int = 42,
                            use_full_preprocessing: bool = True) -> Dict[str, FederatedModel]:
        """
        Train individual models for each region using SAME preprocessing as centralized.
        
        Args:
            regional_datasets: Dictionary mapping region names to datasets
            target_column: Name of target variable column
            validation_split: Fraction of data for validation
            random_state: Random seed for reproducibility
            use_full_preprocessing: Whether to use full preprocessing pipeline
            
        Returns:
            Dictionary of trained federated models
        """
        logger.info(f"Training models for {len(regional_datasets)} regions with FULL preprocessing")
        
        regional_models = {}
        training_summary = []
        
        for region_name, dataset in regional_datasets.items():
            logger.info(f"Training model for {region_name}...")
            start_time = time.time()
            
            try:
                # Prepare data - use SAME preprocessing as centralized
                if target_column not in dataset.columns:
                    logger.error(f"Target column '{target_column}' not found in {region_name} dataset")
                    continue
                
                # Prepare features and target (same as centralized)
                y = dataset[target_column].copy()
                
                # Remove non-feature columns
                columns_to_drop = [target_column, 'candidate_residence', 'city', 'region']
                feature_columns = [col for col in dataset.columns if col not in columns_to_drop]
                X = dataset[feature_columns].copy()
                
                # Handle non-numeric columns
                for col in X.columns:
                    if X[col].dtype == 'object':
                        logger.warning(f"Dropping non-numeric column '{col}' from {region_name}")
                        X = X.drop(columns=[col])
                
                # Fill missing values
                X = X.fillna(X.median())
                
                if len(X) == 0 or len(y.unique()) < 2:
                    logger.warning(f"Insufficient data for {region_name}: {len(X)} samples, {len(y.unique())} classes")
                    continue
                
                logger.info(f"{region_name} raw data: {len(X)} samples, {X.shape[1]} features")
                
                # Apply FULL preprocessing pipeline (same as centralized)
                if use_full_preprocessing:
                    X_processed, y_processed, preprocessing_objects = self._apply_full_preprocessing(
                        X, y, validation_split, random_state
                    )
                else:
                    # Fallback to simple preprocessing
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=validation_split, stratify=y, random_state=random_state
                    )
                    X_processed = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
                    preprocessing_objects = {}
                
                logger.info(f"{region_name} processed: {X_processed['X_train'].shape[0]} train, {X_processed['X_test'].shape[0]} test")
                
                # Train model with SAME parameters as centralized
                model = self.base_model_class(**self.model_params)
                model.fit(X_processed['X_train'], X_processed['y_train'])
                
                # Apply probability calibration (same as centralized)
                if use_full_preprocessing:
                    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                    calibrated_model.fit(X_processed['X_train'], X_processed['y_train'])
                    final_model = calibrated_model
                else:
                    final_model = model
                
                # Evaluate on validation set
                val_predictions = final_model.predict(X_processed['X_test'])
                val_probabilities = final_model.predict_proba(X_processed['X_test'])[:, 1] if hasattr(final_model, 'predict_proba') else val_predictions
                
                metrics = {
                    'accuracy': accuracy_score(X_processed['y_test'], val_predictions),
                    'precision': precision_score(X_processed['y_test'], val_predictions),
                    'recall': recall_score(X_processed['y_test'], val_predictions),
                    'f1_score': f1_score(X_processed['y_test'], val_predictions),
                    'roc_auc': roc_auc_score(X_processed['y_test'], val_probabilities) if len(np.unique(X_processed['y_test'])) > 1 else 0.5
                }
                
                training_time = time.time() - start_time
                
                # Create federated model with preprocessing objects
                fed_model = FederatedModel(
                    region_name=region_name,
                    model=final_model,
                    training_samples=len(X_processed['X_train']),
                    model_params=self.model_params,
                    preprocessing_objects=preprocessing_objects
                )
                fed_model.training_time = training_time
                fed_model.metrics = metrics
                
                regional_models[region_name] = fed_model
                
                # Log success
                logger.info(f"{region_name}: F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}, Time={training_time:.2f}s")
                
                training_summary.append({
                    'Region': region_name,
                    'Training_Samples': len(X_processed['X_train']),
                    'Validation_Samples': len(X_processed['X_test']),
                    'F1_Score': metrics['f1_score'],
                    'ROC_AUC': metrics['roc_auc'],
                    'Training_Time_s': training_time
                })
                
            except Exception as e:
                logger.error(f"Failed to train model for {region_name}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        self.regional_models = regional_models
        
        # Save training summary
        summary_df = pd.DataFrame(training_summary)
        logger.info(f"Successfully trained {len(regional_models)} regional models")
        if len(summary_df) > 0:
            logger.info(f"Average F1-score: {summary_df['F1_Score'].mean():.4f}")
            logger.info(f"Average ROC-AUC: {summary_df['ROC_AUC'].mean():.4f}")
        
        return regional_models

    def _apply_full_preprocessing(self, X, y, validation_split, random_state):
        """
        Apply the SAME preprocessing pipeline as centralized model.
        
        Returns:
            Processed data dictionary and preprocessing objects
        """
        # Step 1: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=random_state
        )
        
        # Step 2: RobustScaler normalization
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 3: SelectKBest feature selection
        k_features = min(50, X_train_scaled.shape[1])  # Same as centralized
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Step 4: SMOTE oversampling (same as centralized)
        smote = SMOTE(random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train_selected, y_train)
        
        logger.info(f"Preprocessing: {X_train.shape[0]} â†’ {X_train_final.shape[0]} samples, {X_train.shape[1]} â†’ {X_train_final.shape[1]} features")
        
        processed_data = {
            'X_train': X_train_final,
            'X_test': X_test_selected,
            'y_train': y_train_final,
            'y_test': y_test
        }
        
        preprocessing_objects = {
            'scaler': scaler,
            'selector': selector
        }
        
        return processed_data, y_train_final, preprocessing_objects

    def aggregate_models(self, method: str = 'voting_ensemble') -> BaseEstimator:
        """
        Aggregate regional models into a global model.
        
        Args:
            method: Aggregation method ('voting_ensemble', 'best_regional', 'weighted_ensemble')
            
        Returns:
            Aggregated global model
        """
        if not self.regional_models:
            logger.error("No regional models available for aggregation")
            return None
        
        logger.info(f"Aggregating {len(self.regional_models)} models using {method}")
        
        if method == 'voting_ensemble':
            return self._create_voting_ensemble()
        elif method == 'best_regional':
            return self._select_best_regional_model()
        elif method == 'weighted_ensemble':
            return self._create_weighted_ensemble()
        else:
            logger.error(f"Unknown aggregation method: {method}")
            return None

    def _create_voting_ensemble(self) -> VotingClassifier:
        """Create voting ensemble from regional models."""
        try:
            estimators = [(region, model.model) for region, model in self.regional_models.items()]
            
            # Determine voting type based on model capabilities
            voting_type = 'soft' if all(hasattr(model.model, 'predict_proba') for model in self.regional_models.values()) else 'hard'
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting_type,
                n_jobs=4
            )
            
            logger.info(f"Created {voting_type} voting ensemble with {len(estimators)} regional models")
            return ensemble
            
        except Exception as e:
            logger.error(f"Failed to create voting ensemble: {str(e)}")
            return None

    def _select_best_regional_model(self) -> BaseEstimator:
        """Select the best performing regional model as global model."""
        best_f1 = 0
        best_model = None
        best_region = None
        
        for region, model in self.regional_models.items():
            f1_score = model.metrics.get('f1_score', 0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model.model
                best_region = region
        
        logger.info(f"Selected best regional model from {best_region} (F1: {best_f1:.4f})")
        return deepcopy(best_model)

    def _create_weighted_ensemble(self) -> BaseEstimator:
        """Create weighted ensemble based on regional performance."""
        # For simplicity, use performance-weighted selection
        # In practice, this would involve sophisticated model parameter averaging
        
        weights = {}
        for region, model in self.regional_models.items():
            f1_score = model.metrics.get('f1_score', 0)
            sample_count = model.training_samples
            # Weight by both performance and sample size
            weights[region] = f1_score * np.log(sample_count + 1)
        
        best_region = max(weights.keys(), key=lambda k: weights[k])
        logger.info(f"Weighted ensemble represented by {best_region} (weight: {weights[best_region]:.4f})")
        
        return deepcopy(self.regional_models[best_region].model)

    def evaluate_federated_vs_centralized(self, centralized_model: BaseEstimator,
                                        test_datasets: Dict[str, pd.DataFrame],
                                        target_column: str = 'outcome') -> pd.DataFrame:
        """
        Compare federated model performance against centralized model with FAIR preprocessing.
        
        Args:
            centralized_model: Trained centralized model (with preprocessing wrapper)
            test_datasets: Test datasets for each region
            target_column: Name of target variable
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Evaluating federated vs centralized models with fair comparison")
        
        comparison_results = []
        
        for region, test_data in test_datasets.items():
            if region not in self.regional_models:
                logger.warning(f"No regional model for {region}")
                continue
                
            try:
                # Prepare test data - SAME preprocessing as training
                columns_to_drop = [target_column, 'candidate_residence', 'city', 'region']
                feature_columns = [col for col in test_data.columns if col not in columns_to_drop]
                
                X_test_raw = test_data[feature_columns].copy()
                y_test = test_data[target_column].copy()
                
                # Remove any non-numeric columns
                for col in X_test_raw.columns:
                    if X_test_raw[col].dtype == 'object':
                        X_test_raw = X_test_raw.drop(columns=[col])
                
                # Fill missing values
                X_test = X_test_raw.fillna(X_test_raw.median())
                
                if len(y_test.unique()) < 2:
                    logger.warning(f"Insufficient test data variance for {region}")
                    continue
                
                logger.info(f"Evaluating {region}: {X_test.shape[0]} samples, {X_test.shape[1]} features")
                
                # Evaluate regional model (with its own preprocessing)
                regional_model = self.regional_models[region]
                regional_pred = regional_model.predict(X_test)
                regional_prob = regional_model.predict_proba(X_test)[:, 1]
                
                # Evaluate centralized model (with its preprocessing wrapper)
                centralized_pred = centralized_model.predict(X_test)
                centralized_prob = centralized_model.predict_proba(X_test)[:, 1]
                
                # Evaluate federated model (use best performing regional as proxy)
                if self.global_model:
                    # If we have a global federated model, use it
                    federated_pred = self.global_model.predict(X_test)
                    federated_prob = self.global_model.predict_proba(X_test)[:, 1] if hasattr(self.global_model, 'predict_proba') else federated_pred
                else:
                    # Use best regional model as federated representative
                    best_regional_model = self._select_best_regional_model()
                    federated_pred = best_regional_model.predict(X_test)
                    federated_prob = best_regional_model.predict_proba(X_test)[:, 1] if hasattr(best_regional_model, 'predict_proba') else federated_pred
                
                # Calculate metrics
                result = {
                    'Region': region,
                    'Test_Samples': len(y_test),
                    
                    # Regional model metrics
                    'Regional_F1': f1_score(y_test, regional_pred),
                    'Regional_Accuracy': accuracy_score(y_test, regional_pred),
                    'Regional_ROC_AUC': roc_auc_score(y_test, regional_prob),
                    
                    # Centralized model metrics
                    'Centralized_F1': f1_score(y_test, centralized_pred),
                    'Centralized_Accuracy': accuracy_score(y_test, centralized_pred),
                    'Centralized_ROC_AUC': roc_auc_score(y_test, centralized_prob),
                    
                    # Federated model metrics
                    'Federated_F1': f1_score(y_test, federated_pred),
                    'Federated_Accuracy': accuracy_score(y_test, federated_pred),
                    'Federated_ROC_AUC': roc_auc_score(y_test, federated_prob)
                }
                
                comparison_results.append(result)
                logger.info(f"{region}: Regional F1={result['Regional_F1']:.4f}, Centralized F1={result['Centralized_F1']:.4f}, Federated F1={result['Federated_F1']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {region}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if len(comparison_df) > 0:
            # Calculate average performance
            avg_metrics = {
                'Region': 'AVERAGE',
                'Test_Samples': comparison_df['Test_Samples'].sum(),
                'Regional_F1': comparison_df['Regional_F1'].mean(),
                'Regional_Accuracy': comparison_df['Regional_Accuracy'].mean(),
                'Regional_ROC_AUC': comparison_df['Regional_ROC_AUC'].mean(),
                'Centralized_F1': comparison_df['Centralized_F1'].mean(),
                'Centralized_Accuracy': comparison_df['Centralized_Accuracy'].mean(),
                'Centralized_ROC_AUC': comparison_df['Centralized_ROC_AUC'].mean(),
                'Federated_F1': comparison_df['Federated_F1'].mean(),
                'Federated_Accuracy': comparison_df['Federated_Accuracy'].mean(),
                'Federated_ROC_AUC': comparison_df['Federated_ROC_AUC'].mean()
            }
            
            comparison_df = pd.concat([comparison_df, pd.DataFrame([avg_metrics])], ignore_index=True)
        
        return comparison_df.round(4)

    def save_federated_models(self, save_dir: str = 'results_federated') -> None:
        """
        Save all regional models and metadata with preprocessing objects.
        
        Args:
            save_dir: Directory to save federated models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual regional models
        models_dir = os.path.join(save_dir, 'regional_models')
        os.makedirs(models_dir, exist_ok=True)
        
        regional_metrics = []
        
        for region, fed_model in self.regional_models.items():
            # Save model with preprocessing
            model_path = os.path.join(models_dir, f"{region}_model.joblib")
            joblib.dump(fed_model, model_path)  # Save entire FederatedModel object
            
            # Save metadata
            metadata = {
                'region': region,
                'training_samples': fed_model.training_samples,
                'training_time': fed_model.training_time,
                'model_params': fed_model.model_params,
                **fed_model.metrics
            }
            regional_metrics.append(metadata)
            
            logger.info(f"Saved {region} model with preprocessing to {model_path}")
        
        # Save global model if available
        if self.global_model:
            global_path = os.path.join(save_dir, 'federated_global_model.joblib')
            joblib.dump(self.global_model, global_path)
            logger.info(f"Saved global federated model to {global_path}")
        
        # Save regional metrics
        metrics_df = pd.DataFrame(regional_metrics)
        metrics_path = os.path.join(save_dir, 'federated_regional_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved regional metrics to {metrics_path}")

    def load_federated_models(self, load_dir: str = 'results_federated') -> bool:
        """
        Load previously saved federated models with preprocessing.
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            True if models loaded successfully, False otherwise
        """
        if not os.path.exists(load_dir):
            logger.error(f"Load directory {load_dir} does not exist")
            return False
        
        models_dir = os.path.join(load_dir, 'regional_models')
        
        # Load regional models
        loaded_models = {}
        
        for region_name in ['CPI_Verona', 'CPI_Vicenza', 'CPI_Padova', 'CPI_Treviso', 'CPI_Venezia']:
            model_path = os.path.join(models_dir, f"{region_name}_model.joblib")
            if os.path.exists(model_path):
                try:
                    fed_model = joblib.load(model_path)  # Load entire FederatedModel object
                    loaded_models[region_name] = fed_model
                    logger.info(f"Loaded {region_name} model with preprocessing from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {region_name} model: {str(e)}")
        
        # Load global model if available
        global_path = os.path.join(load_dir, 'federated_global_model.joblib')
        if os.path.exists(global_path):
            try:
                self.global_model = joblib.load(global_path)
                logger.info(f"Loaded global federated model from {global_path}")
            except Exception as e:
                logger.error(f"Failed to load global model: {str(e)}")
        
        self.regional_models = loaded_models
        return len(loaded_models) > 0


def main():
    """
    Test function for federated trainer.
    """
    logger.info("Testing FederatedTrainer...")
    
    # Test with improved preprocessing pipeline
    trainer = FederatedTrainer(
        base_model_class=lgb.LGBMClassifier,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42
    )
    
    logger.info("FederatedTrainer test completed")


if __name__ == "__main__":
    main()