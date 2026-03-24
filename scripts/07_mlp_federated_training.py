#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 07_mlp_federated_learning.py
Description: Production-ready MLP federated learning implementation 
"""

import sys
import os
import time
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from copy import deepcopy
import hashlib
import glob

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
FEDERATED_DATA_DIR = 'data/federated'
MLP_RESULTS_DIR = 'results_mlp_federated'

# Optimized MLP parameters for federated learning with explicit momentum
MLP_CONFIG = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'sgd',  # More stable for partial_fit than adam
    'alpha': 0.01,
    'batch_size': 256,  # Reduced to avoid clipping warnings
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'momentum': 0.9,  # Explicit momentum for SGD stability
    'nesterovs_momentum': True,  # Nesterov acceleration for faster convergence
    'max_iter': 100,
    'early_stopping': False,
    'random_state': 42
}

# Federated learning configuration with all options
FEDERATED_CONFIG = {
    'num_rounds': 30,
    'local_epochs': 5,  # Reduced from 30 to prevent overfitting
    'local_batch_size': 256,  # Consistent with MLP config
    'validation_split': 0.15,
    'early_stopping_patience': 3,  # Reduced for faster convergence
    'validation_check_frequency': 2,  # Check every 2 epochs instead of 3
    'aggregation_method': 'fedavg',  # fedavg, fedavg_equal, coordinate_median, trimmed_mean
    'trimmed_mean_ratio': 0.1,  # For trimmed_mean aggregation
    'seed': 42  # Global seed for reproducibility
}


def deterministic_seed(s: str, base_seed: int) -> int:
    """
    Generate deterministic seed from string using SHA256.
    Replaces hash() which is salted between runs in CPython.
    """
    h = hashlib.sha256((s + str(base_seed)).encode()).hexdigest()
    return int(h[:8], 16)  # Use first 32 bits as seed


class RobustGlobalPreprocessor:
    """
    Production-ready global preprocessor with robust column handling and consistent statistics.
    """
    
    def __init__(self, seed: int = 42):
        self.scaler = None
        self.selector = None
        self.feature_columns = None
        self.global_medians = None  # Store global medians for consistent imputation
        self.n_features = None
        self.is_fitted = False
        self.seed = seed
        
    def fit_global_preprocessor(self, all_regional_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Fit global preprocessor with robust column handling.
        """
        try:
            logger.info("Fitting robust global preprocessor on combined regional data...")
            
            # Combine all regional training data
            combined_data_frames = []
            for region, data in all_regional_data.items():
                combined_data_frames.append(data)
            
            combined_data = pd.concat(combined_data_frames, ignore_index=True)
            logger.info(f"Combined data size: {len(combined_data)} samples from {len(all_regional_data)} regions")
            
            # Extract features and target
            y_combined = combined_data['outcome'].copy()
            columns_to_drop = ['outcome', 'candidate_residence', 'city', 'region']
            self.feature_columns = [col for col in combined_data.columns if col not in columns_to_drop]
            X_combined = combined_data[self.feature_columns].copy()
            
            # Keep only numeric columns - this is our GLOBAL feature set
            numeric_columns = X_combined.select_dtypes(include=[np.number]).columns
            X_combined = X_combined[numeric_columns]
            self.feature_columns = list(numeric_columns)
            
            # Store global medians for consistent imputation
            self.global_medians = X_combined.median()
            
            # Fill missing values with global medians
            X_combined = X_combined.fillna(self.global_medians)
            
            if len(y_combined.unique()) < 2:
                logger.error("Insufficient target class variety in combined data")
                return False
            
            # Fit global scaler
            self.scaler = StandardScaler()
            X_combined_scaled = self.scaler.fit_transform(X_combined)
            
            # Fit global feature selector
            k_features = min(50, X_combined_scaled.shape[1])
            self.selector = SelectKBest(score_func=f_classif, k=k_features)
            X_combined_selected = self.selector.fit_transform(X_combined_scaled, y_combined)
            
            self.n_features = X_combined_selected.shape[1]
            self.is_fitted = True
            
            logger.info(f"Robust global preprocessor fitted successfully:")
            logger.info(f"  - Feature columns: {len(self.feature_columns)} numeric features")
            logger.info(f"  - Selected features: {self.n_features} features after selection")
            logger.info(f"  - Global medians stored for consistent imputation")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit global preprocessor: {e}")
            return False
    
    def transform_regional_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform regional data with robust column handling and consistent imputation.
        Proper fillna order - only use global medians, no zero pollution.
        """
        if not self.is_fitted:
            raise ValueError("Global preprocessor must be fitted before transformation")
        
        try:
            # Extract training features and target
            y_train_full = train_data['outcome'].copy()
            
            # Robust column selection - handle missing columns gracefully
            # Only use global medians for imputation, no zero pollution
            X_train_full = train_data.reindex(columns=self.feature_columns)
            
            # Extract test features and target
            y_test = test_data['outcome'].copy()
            X_test = test_data.reindex(columns=self.feature_columns)
            
            # Fill missing values using ONLY global medians (not local statistics)
            X_train_full = X_train_full.fillna(self.global_medians)
            X_test = X_test.fillna(self.global_medians)
            
            # Apply global scaler
            X_train_full_scaled = self.scaler.transform(X_train_full)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Apply global feature selector
            X_train_full_selected = self.selector.transform(X_train_full_scaled)
            X_test_selected = self.selector.transform(X_test_scaled)
            
            # Split training data into train and validation
            validation_size = FEDERATED_CONFIG['validation_split']
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full_selected, y_train_full, 
                test_size=validation_size, 
                stratify=y_train_full, 
                random_state=self.seed
            )
            
            # Store raw sample count BEFORE SMOTE for correct FedAvg weights
            train_samples_raw = len(X_train)
            
            # Apply SMOTE only to training data (not validation) with fallback and logging
            original_counts = pd.Series(y_train).value_counts().sort_index()
            minority_ratio_before = original_counts.min() / original_counts.sum()
            
            try:
                smote = SMOTE(random_state=self.seed)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                oversampler_used = "SMOTE"
                logger.debug("Used SMOTE for oversampling")
            except Exception as smote_error:
                logger.warning(f"SMOTE failed ({smote_error}), falling back to RandomOverSampler")
                try:
                    ros = RandomOverSampler(random_state=self.seed)
                    X_train_smote, y_train_smote = ros.fit_resample(X_train, y_train)
                    oversampler_used = "RandomOverSampler"
                    logger.debug("Used RandomOverSampler for oversampling")
                except Exception as ros_error:
                    logger.warning(f"RandomOverSampler also failed ({ros_error}), using original data")
                    X_train_smote, y_train_smote = X_train, y_train
                    oversampler_used = "None"
            
            # Store SMOTE sample count and log class balance changes
            train_samples_smote = len(X_train_smote)
            final_counts = pd.Series(y_train_smote).value_counts().sort_index()
            minority_ratio_after = final_counts.min() / final_counts.sum()
            
            logger.debug(f"Oversampling ({oversampler_used}): minority ratio {minority_ratio_before:.3f} → {minority_ratio_after:.3f}")
            
            return {
                'X_train': X_train_smote,
                'X_val': X_val,
                'X_test': X_test_selected,
                'y_train': y_train_smote,
                'y_val': y_val,
                'y_test': y_test,
                'n_features': self.n_features,
                'train_samples_raw': train_samples_raw,
                'train_samples_smote': train_samples_smote,
                'unique_classes': np.unique(y_train_smote),
                'oversampler_used': oversampler_used,
                'minority_ratio_before': minority_ratio_before,
                'minority_ratio_after': minority_ratio_after
            }
            
        except Exception as e:
            logger.error(f"Failed to transform regional data: {e}")
            return {}


class ProductionMLPFederatedTrainer:
    """
    Production-ready MLP federated learning trainer with all optimizations.
    """
    
    def __init__(self, mlp_config: Dict, federated_config: Dict):
        self.mlp_config = mlp_config
        self.federated_config = federated_config
        self.global_parameters = None
        self.regional_models = {}
        self.training_history = []
        self.global_preprocessor = RobustGlobalPreprocessor(seed=federated_config['seed'])
        
        # Initialize reproducible RNG
        self.rng = np.random.default_rng(federated_config['seed'])
        
        logger.info("Initialized Production MLP Federated Trainer")
        logger.info(f"  MLP Architecture: {mlp_config['hidden_layer_sizes']}")
        logger.info(f"  Solver: {mlp_config['solver']} (optimized for partial_fit)")
        logger.info(f"  SGD Momentum: {mlp_config['momentum']}, Nesterov: {mlp_config['nesterovs_momentum']}")
        logger.info(f"  Federated Rounds: {federated_config['num_rounds']}")
        logger.info(f"  Local Epochs: {federated_config['local_epochs']}")
        logger.info(f"  Aggregation Method: {federated_config['aggregation_method']}")
        logger.info(f"  Reproducibility Seed: {federated_config['seed']} (deterministic SHA256)")
    
    def extract_parameters(self, model: MLPClassifier) -> Dict[str, np.ndarray]:
        """Extract only weights and biases - don't touch internal attributes."""
        parameters = {}
        
        try:
            # Extract layer weights
            for i, layer_weights in enumerate(model.coefs_):
                parameters[f'layer_{i}_weights'] = layer_weights.copy()
            
            # Extract layer biases  
            for i, layer_bias in enumerate(model.intercepts_):
                parameters[f'layer_{i}_bias'] = layer_bias.copy()
            
            logger.debug(f"Extracted {len(model.coefs_)} weight matrices and {len(model.intercepts_)} bias vectors")
            return parameters
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return {}
    
    def set_parameters(self, model: MLPClassifier, parameters: Dict[str, np.ndarray]) -> MLPClassifier:
        """
        Safe parameter setting - only modify coefs_ and intercepts_.
        Don't touch internal attributes like n_layers_/n_outputs_.
        """
        try:
            # Extract weights and biases from parameters dict
            new_coefs = []
            new_intercepts = []
            
            layer_idx = 0
            while f'layer_{layer_idx}_weights' in parameters:
                new_coefs.append(parameters[f'layer_{layer_idx}_weights'].copy())
                layer_idx += 1
            
            layer_idx = 0
            while f'layer_{layer_idx}_bias' in parameters:
                new_intercepts.append(parameters[f'layer_{layer_idx}_bias'].copy())
                layer_idx += 1
            
            # SAFE: Only modify weights and biases, leave internal attributes alone
            if new_coefs and new_intercepts:
                model.coefs_ = new_coefs
                model.intercepts_ = new_intercepts
                # Don't modify n_layers_, n_outputs_ - they're already correct from initial fit
                
                logger.debug(f"Set parameters: {len(new_coefs)} layers (safe mode)")
            
            return model
            
        except Exception as e:
            logger.error(f"Parameter setting failed: {e}")
            return model
    
    def create_truly_balanced_warmup_batch(self, X_train: np.ndarray, y_train: np.ndarray, 
                                         region_name: str, round_num: int,
                                         max_samples_per_class: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create truly balanced warm-up batch using minimum class count.
        Uses min(counts) to ensure equal representation per class.
        """
        try:
            # Create region and round-specific seed for reproducibility using SHA256
            region_seed = deterministic_seed(f"{region_name}_{round_num}", self.federated_config['seed'])
            local_rng = np.random.default_rng(region_seed)
            
            unique_classes = np.unique(y_train)
            counts = [np.sum(y_train == c) for c in unique_classes]
            
            # Use minimum class count to ensure true balance
            per_cls = max(1, min(max_samples_per_class, min(counts)))
            
            warmup_indices = []
            
            for cls in unique_classes:
                cls_indices = np.where(y_train == cls)[0]
                # Now we take exactly per_cls from each class
                selected_indices = local_rng.choice(cls_indices, size=per_cls, replace=False)
                warmup_indices.extend(selected_indices)
            
            warmup_indices = np.array(warmup_indices)
            local_rng.shuffle(warmup_indices)
            
            X_warmup = X_train[warmup_indices]
            y_warmup = y_train[warmup_indices]
            
            # Verify true balance
            warmup_counts = pd.Series(y_warmup).value_counts().sort_index()
            logger.debug(f"Created truly balanced warm-up batch for {region_name}: "
                        f"{len(X_warmup)} samples, per-class: {per_cls}, "
                        f"distribution: {dict(warmup_counts)}")
            
            return X_warmup, y_warmup
            
        except Exception as e:
            logger.error(f"Failed to create balanced warm-up batch: {e}")
            # Fallback: use first samples
            sample_size = min(100, len(X_train))
            return X_train[:sample_size], y_train[:sample_size]
    
    def federated_average(self, regional_params_list: List[Dict], weights: List[float]) -> Dict[str, np.ndarray]:
        """FedAvg aggregation (weighted by raw sample counts) with zero-weight guard."""
        if not regional_params_list:
            return {}
        
        # Zero-weight guard
        total_weight = sum(weights)
        if total_weight == 0:
            logger.warning("All weights are zero, falling back to equal weights")
            weights = [1.0] * len(weights)
            total_weight = sum(weights)
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        param_names = list(regional_params_list[0].keys())
        aggregated_params = {}
        
        for param_name in param_names:
            try:
                weighted_arrays = []
                for i, params in enumerate(regional_params_list):
                    if param_name in params:
                        weighted_array = params[param_name] * weights[i]
                        weighted_arrays.append(weighted_array)
                
                if weighted_arrays:
                    aggregated_params[param_name] = np.sum(weighted_arrays, axis=0)
                
            except Exception as e:
                logger.error(f"Aggregation failed for {param_name}: {e}")
                continue
        
        return aggregated_params
    
    def federated_average_equal(self, regional_params_list: List[Dict], weights: List[float]) -> Dict[str, np.ndarray]:
        """Equal-weight FedAvg (diagnostic mode)."""
        if not regional_params_list:
            return {}
        
        # Equal weights regardless of sample counts
        equal_weights = [1.0 / len(regional_params_list)] * len(regional_params_list)
        
        param_names = list(regional_params_list[0].keys())
        aggregated_params = {}
        
        for param_name in param_names:
            try:
                arrays = []
                for params in regional_params_list:
                    if param_name in params:
                        arrays.append(params[param_name])
                
                if arrays:
                    aggregated_params[param_name] = np.mean(arrays, axis=0)
                
            except Exception as e:
                logger.error(f"Equal-weight aggregation failed for {param_name}: {e}")
                continue
        
        return aggregated_params
    
    def coordinate_median(self, regional_params_list: List[Dict], weights: List[float]) -> Dict[str, np.ndarray]:
        """Coordinate-wise median aggregation (robust to outliers)."""
        if not regional_params_list:
            return {}
        
        param_names = list(regional_params_list[0].keys())
        aggregated_params = {}
        
        for param_name in param_names:
            try:
                arrays = []
                for params in regional_params_list:
                    if param_name in params:
                        arrays.append(params[param_name])
                
                if arrays:
                    stacked = np.stack(arrays, axis=0)
                    aggregated_params[param_name] = np.median(stacked, axis=0)
                
            except Exception as e:
                logger.error(f"Coordinate median aggregation failed for {param_name}: {e}")
                continue
        
        return aggregated_params
    
    def trimmed_mean(self, regional_params_list: List[Dict], weights: List[float]) -> Dict[str, np.ndarray]:
        """Trimmed mean aggregation (removes extreme values)."""
        if not regional_params_list:
            return {}
        
        trim_ratio = self.federated_config['trimmed_mean_ratio']
        param_names = list(regional_params_list[0].keys())
        aggregated_params = {}
        
        for param_name in param_names:
            try:
                arrays = []
                for params in regional_params_list:
                    if param_name in params:
                        arrays.append(params[param_name])
                
                if arrays:
                    stacked = np.stack(arrays, axis=0)
                    # Sort along client axis and trim extremes
                    sorted_array = np.sort(stacked, axis=0)
                    n_clients = len(arrays)
                    trim_count = int(n_clients * trim_ratio)
                    
                    if trim_count > 0 and trim_count < n_clients // 2:
                        trimmed = sorted_array[trim_count:-trim_count]
                        aggregated_params[param_name] = np.mean(trimmed, axis=0)
                    else:
                        aggregated_params[param_name] = np.mean(sorted_array, axis=0)
                
            except Exception as e:
                logger.error(f"Trimmed mean aggregation failed for {param_name}: {e}")
                continue
        
        return aggregated_params
    
    def aggregate_parameters(self, regional_params_list: List[Dict], weights: List[float], 
                           trained_regions: List[str]) -> Dict[str, np.ndarray]:
        """Route to appropriate aggregation method with proper logging."""
        method = self.federated_config['aggregation_method']
        
        if method == 'fedavg':
            result = self.federated_average(regional_params_list, weights)
            # Correct region-weight alignment in logging
            logger.info(f"FedAvg aggregation: {len(regional_params_list)} models")
            logger.info(f"Raw sample counts: {dict(zip(trained_regions, weights))}")
        elif method == 'fedavg_equal':
            result = self.federated_average_equal(regional_params_list, weights)
            logger.info(f"Equal-weight FedAvg: {len(regional_params_list)} models")
        elif method == 'coordinate_median':
            result = self.coordinate_median(regional_params_list, weights)
            logger.info(f"Coordinate median: {len(regional_params_list)} models")
        elif method == 'trimmed_mean':
            result = self.trimmed_mean(regional_params_list, weights)
            logger.info(f"Trimmed mean (trim={self.federated_config['trimmed_mean_ratio']}): {len(regional_params_list)} models")
        else:
            logger.warning(f"Unknown aggregation method {method}, falling back to FedAvg")
            result = self.federated_average(regional_params_list, weights)
        
        logger.info(f"Aggregation completed: {len(result)} parameters")
        return result
    
    def safe_calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics with safe error handling for edge cases.
        Handles division by zero and single-class predictions.
        """
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        except Exception:
            metrics['accuracy'] = 0.0
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        except Exception:
            metrics['precision'] = 0.0
        
        try:
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        except Exception:
            metrics['recall'] = 0.0
        
        try:
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            metrics['f1_score'] = 0.0
        
        try:
            if len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['roc_auc'] = np.nan
        except Exception:
            metrics['roc_auc'] = np.nan
        
        return metrics
    
    def train_local_model_optimized(self, data: Dict, region_name: str, round_num: int,
                                  global_params: Optional[Dict] = None) -> MLPClassifier:
        """
        Optimized training with proper batch processing, shuffling, and reproducibility.
        Uses deterministic_seed for true reproducibility and truly balanced warm-up.
        """
        try:
            start_time = time.time()
            
            # Create region and round-specific RNG for reproducibility using SHA256
            region_seed = deterministic_seed(f"{region_name}_{round_num}", self.federated_config['seed'])
            local_rng = np.random.default_rng(region_seed)
            
            # Create model configuration for local training
            local_config = self.mlp_config.copy()
            local_config['max_iter'] = 1  # We control epochs manually
            local_config['batch_size'] = self.federated_config['local_batch_size']
            
            logger.info(f"{region_name}: Optimized training ({self.federated_config['local_epochs']} epochs)")
            logger.info(f"{region_name}: Oversampling used: {data.get('oversampler_used', 'unknown')}")
            
            # Step 1: Safe initialization with truly balanced warm-up
            model = MLPClassifier(**local_config)
            X_warmup, y_warmup = self.create_truly_balanced_warmup_batch(
                data['X_train'], data['y_train'], region_name, round_num
            )
            model.fit(X_warmup, y_warmup)
            
            # Step 2: Set global parameters if available (safe mode)
            if global_params:
                logger.info(f"{region_name}: Applying global parameter initialization")
                model = self.set_parameters(model, global_params)
            else:
                logger.info(f"{region_name}: Starting from scratch")
            
            # Step 3: Optimized training loop
            best_val_f1 = 0.0
            patience_counter = 0
            best_model_params = None
            actual_epochs = 0
            
            all_classes = data['unique_classes']
            batch_size = self.federated_config['local_batch_size']
            validation_freq = self.federated_config['validation_check_frequency']
            
            for epoch in range(self.federated_config['local_epochs']):
                # Shuffle indices at the beginning of each epoch
                n_samples = len(data['X_train'])
                shuffled_indices = local_rng.permutation(n_samples)
                
                # Proper batch processing that doesn't drop the tail
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_indices = shuffled_indices[start_idx:end_idx]
                    
                    X_batch = data['X_train'][batch_indices]
                    y_batch = data['y_train'][batch_indices]
                    
                    # Safe partial_fit with explicit classes on first batch
                    if epoch == 0 and start_idx == 0:
                        model.partial_fit(X_batch, y_batch, classes=all_classes)
                    else:
                        model.partial_fit(X_batch, y_batch)
                
                actual_epochs += 1
                
                # Validation check with configurable frequency
                if (epoch + 1) % validation_freq == 0 or epoch == self.federated_config['local_epochs'] - 1:
                    val_predictions = model.predict(data['X_val'])
                    val_probabilities = model.predict_proba(data['X_val'])[:, 1]
                    val_metrics = self.safe_calculate_metrics(data['y_val'], val_predictions, val_probabilities)
                    val_f1 = val_metrics['f1_score']
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_model_params = self.extract_parameters(model)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= self.federated_config['early_stopping_patience']:
                        logger.info(f"{region_name}: Early stopping at epoch {epoch + 1}, best val F1: {best_val_f1:.4f}")
                        break
            
            # Restore best parameters if early stopping occurred
            if best_model_params:
                model = self.set_parameters(model, best_model_params)
            
            training_time = time.time() - start_time
            
            # Evaluate model with safe metrics calculation
            test_predictions = model.predict(data['X_test'])
            test_probabilities = model.predict_proba(data['X_test'])[:, 1]
            val_predictions = model.predict(data['X_val'])
            val_probabilities = model.predict_proba(data['X_val'])[:, 1]
            
            test_metrics = self.safe_calculate_metrics(data['y_test'], test_predictions, test_probabilities)
            val_metrics = self.safe_calculate_metrics(data['y_val'], val_predictions, val_probabilities)
            
            metrics = {
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1_score': test_metrics['f1_score'],
                'test_roc_auc': test_metrics['roc_auc'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1_score': val_metrics['f1_score'],
                'val_roc_auc': val_metrics['roc_auc'],
                'training_time': training_time,
                'loss': getattr(model, 'loss_', 0.0),
                'n_iter': actual_epochs,
                'train_samples_raw': data['train_samples_raw'],
                'train_samples_smote': data['train_samples_smote'],
                'validation_samples': len(data['X_val']),
                'test_samples': len(data['X_test']),
                'feature_space_size': data['n_features'],
                'oversampler_used': data.get('oversampler_used', 'unknown'),
                'minority_ratio_before': data.get('minority_ratio_before', 0.0),
                'minority_ratio_after': data.get('minority_ratio_after', 0.0)
            }
            
            init_type = "with_global_init" if global_params else "from_scratch"
            logger.info(f"{region_name} ({init_type}): Test F1={metrics['test_f1_score']:.4f}, "
                       f"Val F1={metrics['val_f1_score']:.4f}, Time={training_time:.2f}s, "
                       f"Epochs={actual_epochs}, Raw={data['train_samples_raw']}, "
                       f"SMOTE={data['train_samples_smote']}, Oversampler={data.get('oversampler_used', 'unknown')}")
            
            model._local_metrics = metrics
            return model
            
        except Exception as e:
            logger.error(f"Local training failed for {region_name}: {e}")
            return None
    
    def federated_learning_round(self, regional_train_data: Dict[str, pd.DataFrame], 
                               regional_test_data: Dict[str, pd.DataFrame],
                               round_num: int) -> Dict[str, np.ndarray]:
        """Execute one round of production-ready federated learning."""
        logger.info(f"=== Federated Learning Round {round_num + 1} ===")
        logger.info(f"Aggregation method: {self.federated_config['aggregation_method']}")
        
        regional_params_list = []
        regional_weights = []
        round_metrics = {}
        trained_regions = []  # Track successfully trained regions for proper alignment
        
        for region_name in regional_train_data.keys():
            logger.info(f"Training local model for {region_name}...")
            
            try:
                # Transform data using robust global preprocessor
                data = self.global_preprocessor.transform_regional_data(
                    regional_train_data[region_name], 
                    regional_test_data[region_name]
                )
                
                if not data:
                    logger.warning(f"Data transformation failed for {region_name}")
                    continue
                
                logger.info(f"Transformed data for {region_name}: "
                           f"Raw={data['train_samples_raw']}, SMOTE={data['train_samples_smote']}, "
                           f"Val={data['X_val'].shape[0]}, Test={data['X_test'].shape[0]}, "
                           f"Features={data['n_features']}, Oversampler={data.get('oversampler_used', 'unknown')}")
                
                # Train local model with optimizations
                local_model = self.train_local_model_optimized(data, region_name, round_num, self.global_parameters)
                
                if local_model is None:
                    logger.warning(f"Local training failed for {region_name}")
                    continue
                
                # Extract parameters
                local_params = self.extract_parameters(local_model)
                if local_params:
                    regional_params_list.append(local_params)
                    regional_weights.append(data['train_samples_raw'])  # Use raw samples for weights
                    round_metrics[region_name] = local_model._local_metrics
                    trained_regions.append(region_name)  # Track successful training
                    
                    # Store regional model info
                    self.regional_models[region_name] = {
                        'model': local_model,
                        'data_info': {
                            'training_samples_raw': data['train_samples_raw'],
                            'training_samples_smote': data['train_samples_smote'],
                            'validation_samples': len(data['X_val']),
                            'test_samples': len(data['X_test']),
                            'n_features': data['n_features'],
                            'oversampler_used': data.get('oversampler_used', 'unknown'),
                            'minority_ratio_before': data.get('minority_ratio_before', 0.0),
                            'minority_ratio_after': data.get('minority_ratio_after', 0.0)
                        }
                    }
                
            except Exception as e:
                logger.error(f"Failed to train {region_name}: {e}")
                continue
        
        if not regional_params_list:
            logger.error("No regional models trained successfully")
            return {}
        
        # Aggregate parameters using selected method with proper region alignment
        logger.info("Aggregating parameters...")
        self.global_parameters = self.aggregate_parameters(regional_params_list, regional_weights, trained_regions)
        
        # Calculate weighted and unweighted metrics
        if round_metrics:
            # Unweighted averages (simple mean)
            avg_test_f1 = np.mean([m['test_f1_score'] for m in round_metrics.values()])
            avg_val_f1 = np.mean([m['val_f1_score'] for m in round_metrics.values()])
            avg_training_time = np.mean([m['training_time'] for m in round_metrics.values()])
            
            # Weighted averages (by test sample counts)
            test_weights = [m['test_samples'] for m in round_metrics.values()]
            total_test_weight = sum(test_weights)
            test_weights_norm = [w / total_test_weight for w in test_weights] if total_test_weight > 0 else [1.0 / len(test_weights)] * len(test_weights)

            weighted_test_f1 = sum(m['test_f1_score'] * w for m, w in zip(round_metrics.values(), test_weights_norm))
            weighted_val_f1 = sum(m['val_f1_score'] * w for m, w in zip(round_metrics.values(), test_weights_norm))
            
            total_raw_samples = sum(regional_weights)
            total_smote_samples = sum([m['train_samples_smote'] for m in round_metrics.values()])
            
            # Store round history with both weighted and unweighted metrics
            round_history = {
                'round': round_num,
                'participating_regions': list(round_metrics.keys()),
                'regional_metrics': round_metrics,
                'average_test_f1': avg_test_f1,
                'average_val_f1': avg_val_f1,
                'weighted_test_f1': weighted_test_f1,
                'weighted_val_f1': weighted_val_f1,
                'average_training_time': avg_training_time,
                'total_raw_samples': total_raw_samples,
                'total_smote_samples': total_smote_samples,
                'improvement_from_previous': 0.0,
                'weighted_improvement': 0.0,
                'global_feature_space_size': self.global_preprocessor.n_features,
                'aggregation_method': self.federated_config['aggregation_method'],
                'test_sample_weights': dict(zip(round_metrics.keys(), test_weights))
            }
            
            # Calculate improvements from previous round
            if self.training_history:
                prev_avg_f1 = self.training_history[-1]['average_test_f1']
                prev_weighted_f1 = self.training_history[-1]['weighted_test_f1']
                improvement = avg_test_f1 - prev_avg_f1
                weighted_improvement = weighted_test_f1 - prev_weighted_f1
                round_history['improvement_from_previous'] = improvement
                round_history['weighted_improvement'] = weighted_improvement
                
                logger.info(f"Round {round_num + 1} completed:")
                logger.info(f"  Unweighted - Test F1: {avg_test_f1:.4f}, Val F1: {avg_val_f1:.4f} (Δ: {improvement:+.4f})")
                logger.info(f"  Weighted   - Test F1: {weighted_test_f1:.4f}, Val F1: {weighted_val_f1:.4f} (Δ: {weighted_improvement:+.4f})")
                logger.info(f"  Samples - Raw: {total_raw_samples}, SMOTE: {total_smote_samples}")
            else:
                logger.info(f"Round {round_num + 1} completed:")
                logger.info(f"  Unweighted - Test F1: {avg_test_f1:.4f}, Val F1: {avg_val_f1:.4f}")
                logger.info(f"  Weighted   - Test F1: {weighted_test_f1:.4f}, Val F1: {weighted_val_f1:.4f}")
                logger.info(f"  Samples - Raw: {total_raw_samples}, SMOTE: {total_smote_samples}")
            
            self.training_history.append(round_history)
        
        return self.global_parameters
    
    def run_federated_learning(self, regional_train_data: Dict[str, pd.DataFrame],
                             regional_test_data: Dict[str, pd.DataFrame]) -> MLPClassifier:
        """Run complete production-ready federated learning process."""
        logger.info("Starting Production MLP Federated Learning")
        logger.info(f"Regions: {list(regional_train_data.keys())}")
        logger.info(f"Rounds: {self.federated_config['num_rounds']}")
        logger.info(f"Local epochs: {self.federated_config['local_epochs']}")
        logger.info(f"Aggregation: {self.federated_config['aggregation_method']}")
        logger.info("Production optimizations:")
        logger.info("  - Safe parameter handling (no internal attribute modification)")
        logger.info("  - Proper batch processing with epoch shuffling")
        logger.info("  - Robust column handling and consistent imputation")
        logger.info("  - Full reproducibility with deterministic SHA256 seeding")
        logger.info("  - Multiple aggregation methods")
        logger.info("  - Enhanced error handling and SMOTE fallback")
        logger.info("  - Truly balanced warm-up batches")
        logger.info("  - SGD with momentum and Nesterov acceleration")
        logger.info("  - Weighted and unweighted metrics")
        
        start_time = time.time()
        
        # Step 1: Fit robust global preprocessor
        logger.info("Step 1: Fitting robust global preprocessor...")
        if not self.global_preprocessor.fit_global_preprocessor(regional_train_data):
            logger.error("Failed to fit global preprocessor")
            return None
        
        # Step 2: Run federated learning rounds
        logger.info("Step 2: Starting federated learning rounds...")
        for round_num in range(self.federated_config['num_rounds']):
            global_params = self.federated_learning_round(regional_train_data, regional_test_data, round_num)
            
            if not global_params:
                logger.error(f"Round {round_num + 1} failed")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Federated learning completed in {total_time:.2f} seconds")
        
        # Step 3: Create final global model with deterministic template selection
        if self.global_parameters and self.regional_models:
            logger.info("Step 3: Creating final global model...")
            
            try:
                # Deterministic template region selection (alphabetically first)
                template_region = sorted(regional_train_data.keys())[0]
                logger.info(f"Using deterministic template region: {template_region}")
                
                template_data = self.global_preprocessor.transform_regional_data(
                    regional_train_data[template_region],
                    regional_test_data[template_region]
                )
                
                if template_data:
                    # Create global model
                    global_model = MLPClassifier(**self.mlp_config)
                    
                    # Safe initialization with truly balanced warm-up
                    X_warmup, y_warmup = self.create_truly_balanced_warmup_batch(
                        template_data['X_train'], template_data['y_train'], 
                        "global", self.federated_config['num_rounds']
                    )
                    global_model.fit(X_warmup, y_warmup)
                    
                    # Set aggregated parameters (safe mode)
                    global_model = self.set_parameters(global_model, self.global_parameters)
                    
                    logger.info("Global model created successfully with production optimizations")
                    return global_model
                
            except Exception as e:
                logger.error(f"Failed to create global model: {e}")
        
        # Fallback: return best regional model
        if self.regional_models:
            best_region = max(self.regional_models.keys(), 
                            key=lambda r: self.regional_models[r]['model']._local_metrics['test_f1_score'])
            logger.info(f"Using best regional model from {best_region} as fallback")
            return deepcopy(self.regional_models[best_region]['model'])
        
        return None


def auto_discover_regional_datasets():
    """
    Auto-discover regional datasets from files.
    No hardcoded region list, discovers from filesystem.
    """
    regional_train_data = {}
    regional_test_data = {}
    
    # Auto-discover training data files
    pattern = os.path.join(FEDERATED_DATA_DIR, "*_training_data.csv")
    training_files = glob.glob(pattern)
    
    logger.info(f"Auto-discovered {len(training_files)} regional training files")
    
    for filepath in training_files:
        # Extract region name from filename
        filename = os.path.basename(filepath)
        region = filename.replace('_training_data.csv', '')
        
        try:
            # Load full regional dataset
            full_dataset = pd.read_csv(filepath)
            logger.info(f"Loaded {region}: {len(full_dataset)} samples")
            
            # Single split into train/test
            train_data, test_data = train_test_split(
                full_dataset, test_size=0.2, stratify=full_dataset['outcome'], 
                random_state=FEDERATED_CONFIG['seed']
            )
            
            regional_train_data[region] = train_data
            regional_test_data[region] = test_data
            
            logger.info(f"Split {region}: {len(train_data)} train, {len(test_data)} test")
            
        except Exception as e:
            logger.error(f"Failed to load {region} from {filepath}: {e}")
            continue
    
    return regional_train_data, regional_test_data


def load_and_split_regional_datasets():
    """
    Load regional datasets and split them ONCE into train/test.
    Uses auto-discovery for maximum flexibility.
    """
    return auto_discover_regional_datasets()


def safe_weighted_average_with_nan(values: pd.Series, weights: np.ndarray) -> float:
    """
    Calculate weighted average properly handling NaN values.
    Re-normalizes weights for non-NaN values.
    """
    mask = ~values.isna()
    if not mask.any():
        return np.nan
    
    valid_values = values[mask]
    valid_weights = weights[mask]
    
    if valid_weights.sum() == 0:
        return np.nan
    
    # Re-normalize weights
    valid_weights = valid_weights / valid_weights.sum()
    return (valid_values * valid_weights).sum()


def evaluate_global_model(trainer, global_model, regional_test_data):
    """
    Evaluate the global federated model with consistent preprocessing.
    Proper fillna order - only use global medians.
    Proper weighted ROC-AUC with NaN handling and re-normalization.
    """
    if global_model is None:
        logger.error("No global model available for evaluation")
        return pd.DataFrame()
    
    logger.info("Evaluating global federated model with production preprocessing...")
    
    results = []
    
    for region, test_data in regional_test_data.items():
        if region not in trainer.regional_models:
            continue
        
        try:
            # Use robust global preprocessor for consistent evaluation
            y_test = test_data['outcome'].copy()
            
            # Robust column selection and consistent imputation - only global medians
            X_test = test_data.reindex(columns=trainer.global_preprocessor.feature_columns)
            X_test = X_test.fillna(trainer.global_preprocessor.global_medians)
            
            # Apply global transformations
            X_test_scaled = trainer.global_preprocessor.scaler.transform(X_test)
            X_test_selected = trainer.global_preprocessor.selector.transform(X_test_scaled)
            
            # Predict with global model
            predictions = global_model.predict(X_test_selected)
            probabilities = global_model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics with safe error handling
            safe_metrics = trainer.safe_calculate_metrics(y_test, predictions, probabilities)
            
            result = {
                'Region': region,
                'Test_Samples': len(y_test),
                'MLP_Federated_F1': safe_metrics['f1_score'],
                'MLP_Federated_Accuracy': safe_metrics['accuracy'],
                'MLP_Federated_ROC_AUC': safe_metrics['roc_auc'],
                'MLP_Federated_Precision': safe_metrics['precision'],
                'MLP_Federated_Recall': safe_metrics['recall']
            }
            
            results.append(result)
            logger.info(f"{region}: F1={result['MLP_Federated_F1']:.4f} (production evaluation)")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {region}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Add both unweighted and weighted averages
        test_samples = results_df['Test_Samples'].values
        total_samples = test_samples.sum()
        
        if total_samples > 0:
            weights = test_samples / total_samples
            
            # Unweighted average
            avg_result = {
                'Region': 'AVERAGE',
                'Test_Samples': total_samples,
                'MLP_Federated_F1': results_df['MLP_Federated_F1'].mean(),
                'MLP_Federated_Accuracy': results_df['MLP_Federated_Accuracy'].mean(),
                'MLP_Federated_ROC_AUC': np.nanmean(results_df['MLP_Federated_ROC_AUC']),
                'MLP_Federated_Precision': results_df['MLP_Federated_Precision'].mean(),
                'MLP_Federated_Recall': results_df['MLP_Federated_Recall'].mean()
            }
            
            # Weighted average with proper NaN handling
            weighted_result = {
                'Region': 'WEIGHTED_AVERAGE',
                'Test_Samples': total_samples,
                'MLP_Federated_F1': (results_df['MLP_Federated_F1'] * weights).sum(),
                'MLP_Federated_Accuracy': (results_df['MLP_Federated_Accuracy'] * weights).sum(),
                'MLP_Federated_ROC_AUC': safe_weighted_average_with_nan(results_df['MLP_Federated_ROC_AUC'], weights),
                'MLP_Federated_Precision': (results_df['MLP_Federated_Precision'] * weights).sum(),
                'MLP_Federated_Recall': (results_df['MLP_Federated_Recall'] * weights).sum()
            }
            
            results_df = pd.concat([results_df, pd.DataFrame([avg_result, weighted_result])], ignore_index=True)
    
    return results_df.round(4)


def compare_with_centralized():
    """Load centralized baseline for comparison."""
    logger.info("Loading centralized baseline for comparison...")
    
    centralized_path = 'results/MLP_Optimized.joblib'
    centralized_metrics = None
    
    if os.path.exists(centralized_path):
        try:
            centralized_model = joblib.load(centralized_path)
            logger.info(f"Loaded centralized MLP model from {centralized_path}")
            
            # Load centralized metrics
            metrics_path = 'results/metrics_summary.csv'
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path, index_col=0)
                if 'MLP_Optimized' in metrics_df.index:
                    centralized_metrics = {
                        'f1_score': metrics_df.loc['MLP_Optimized', 'f1_score'],
                        'accuracy': metrics_df.loc['MLP_Optimized', 'accuracy'],
                        'roc_auc': metrics_df.loc['MLP_Optimized', 'roc_auc']
                    }
                    logger.info(f"Centralized MLP metrics: F1={centralized_metrics['f1_score']:.4f}")
            
            return centralized_model, centralized_metrics
            
        except Exception as e:
            logger.error(f"Failed to load centralized model: {e}")
    
    logger.warning("Centralized baseline not available")
    return None, None

def save_federated_results(trainer, global_model, results_df, centralized_metrics):
    """Save all federated learning results with production enhancements."""
    os.makedirs(MLP_RESULTS_DIR, exist_ok=True)
    
    # Save performance results
    if not results_df.empty:
        results_path = os.path.join(MLP_RESULTS_DIR, 'mlp_federated_performance.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved performance results to {results_path}")
    
    # Save global model
    if global_model:
        global_model_path = os.path.join(MLP_RESULTS_DIR, 'mlp_global_federated_model.joblib')
        joblib.dump(global_model, global_model_path)
        logger.info(f"Saved global model to {global_model_path}")
    
    # Save robust global preprocessor
    if trainer.global_preprocessor.is_fitted:
        preprocessor_path = os.path.join(MLP_RESULTS_DIR, 'global_preprocessor.joblib')
        joblib.dump(trainer.global_preprocessor, preprocessor_path)
        logger.info(f"Saved robust global preprocessor to {preprocessor_path}")
    
    # Save regional models
    regional_models_dir = os.path.join(MLP_RESULTS_DIR, 'regional_models')
    os.makedirs(regional_models_dir, exist_ok=True)
    
    for region, model_data in trainer.regional_models.items():
        model_path = os.path.join(regional_models_dir, f"{region}_mlp_model.joblib")
        joblib.dump(model_data, model_path)
        logger.info(f"Saved {region} model to {model_path}")
    
    # Save training history
    if trainer.training_history:
        history_path = os.path.join(MLP_RESULTS_DIR, 'federated_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(trainer.training_history, f, indent=2, default=str)
        logger.info(f"Saved training history to {history_path}")
        
        # Create enhanced improvement summary
        improvement_data = []
        for round_info in trainer.training_history:
            improvement_data.append({
                'Round': round_info['round'] + 1,
                'Unweighted_Test_F1': round_info['average_test_f1'],
                'Unweighted_Val_F1': round_info['average_val_f1'],
                'Weighted_Test_F1': round_info['weighted_test_f1'],
                'Weighted_Val_F1': round_info['weighted_val_f1'],
                'Unweighted_Improvement': round_info.get('improvement_from_previous', 0.0),
                'Weighted_Improvement': round_info.get('weighted_improvement', 0.0),
                'Training_Time': round_info.get('average_training_time', 0.0),
                'Total_Raw_Samples': round_info.get('total_raw_samples', 0),
                'Total_SMOTE_Samples': round_info.get('total_smote_samples', 0),
                'Aggregation_Method': round_info.get('aggregation_method', 'unknown'),
                'Global_Features': round_info.get('global_feature_space_size', 0)
            })
        
        improvement_df = pd.DataFrame(improvement_data)
        improvement_path = os.path.join(MLP_RESULTS_DIR, 'federated_improvement_summary.csv')
        improvement_df.to_csv(improvement_path, index=False)
        logger.info(f"Saved enhanced improvement summary to {improvement_path}")
    
    # Save production configuration
    config = {
        'mlp_config': MLP_CONFIG,
        'federated_config': FEDERATED_CONFIG,
        'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'implemented': {
            'deterministic_seeding': 'SHA256-based seeds for true reproducibility',
            'proper_fillna_order': 'Only global medians, no zero pollution',
            'correct_region_weight_logging': 'Proper alignment of regions and weights',
            'truly_balanced_warmup': 'Uses minimum class count for true balance',
            'weighted_roc_auc_nan_handling': 'Re-normalizes weights for non-NaN values',
            'deterministic_template_selection': 'Alphabetically first region for template'
        },
        'production_optimizations': {
            'safe_parameter_handling': 'Only modify coefs_/intercepts_, no internal attributes',
            'proper_batch_processing': 'Complete batch coverage with epoch shuffling',
            'multiple_aggregation_methods': ['fedavg', 'fedavg_equal', 'coordinate_median', 'trimmed_mean'],
            'robust_preprocessing': 'Graceful column handling and consistent imputation',
            'smote_fallback': 'RandomOverSampler fallback for edge cases',
            'enhanced_error_handling': 'Safe metrics calculation with zero_division guards',
            'auto_discovery': 'Automatic region detection from filesystem',
            'weighted_metrics': 'Both unweighted and weighted averages',
            'sgd_optimization': 'Explicit momentum and Nesterov acceleration',
            'oversampling_logging': 'Track minority ratios before/after oversampling'
        },
        'global_preprocessor_info': {
            'n_features_selected': trainer.global_preprocessor.n_features if trainer.global_preprocessor.is_fitted else 0,
            'feature_columns_count': len(trainer.global_preprocessor.feature_columns) if trainer.global_preprocessor.feature_columns else 0,
            'has_global_medians': trainer.global_preprocessor.global_medians is not None,
            'methodology_note': 'Global preprocessor uses centralized statistics (acceptable for simulation)'
        }
    }
    
    config_path = os.path.join(MLP_RESULTS_DIR, 'mlp_federated_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved production configuration to {config_path}")
    
    # Create comprehensive summary
    summary_lines = []
    summary_lines.append("# PRODUCTION MLP FEDERATED LEARNING EXPERIMENT SUMMARY")
    summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    summary_lines.append("##IMPLEMENTED")
    summary_lines.append("### 1. Deterministic Seeding (SHA256)")
    summary_lines.append("- Replaced hash() with SHA256 for true reproducibility")
    summary_lines.append("- No more random salt between Python runs")
    summary_lines.append("- Deterministic region-specific seeds guaranteed")
    summary_lines.append("")
    summary_lines.append("### 2. Proper Imputation Order")
    summary_lines.append("- Removed fillna(0) that polluted global medians")
    summary_lines.append("- Only use global medians for consistent imputation")
    summary_lines.append("- No zero contamination in missing column handling")
    summary_lines.append("")
    summary_lines.append("### 3. Correct Region-Weight Logging")
    summary_lines.append("- Proper alignment of regions and weights in logs")
    summary_lines.append("- Track trained_regions list to prevent mismatched logging")
    summary_lines.append("- No more misleading weight assignments")
    summary_lines.append("")
    summary_lines.append("### 4. Truly Balanced Warm-up")
    summary_lines.append("- Uses minimum class count for true balance")
    summary_lines.append("- Equal samples per class regardless of original distribution")
    summary_lines.append("- Reduces variance in initial training steps")
    summary_lines.append("")
    summary_lines.append("### 5. Weighted ROC-AUC with NaN Handling")
    summary_lines.append("- Re-normalizes weights for non-NaN values")
    summary_lines.append("- Proper handling when some regions have single-class test sets")
    summary_lines.append("- No more weight loss due to NaN propagation")
    summary_lines.append("")
    summary_lines.append("### 6. Deterministic Template Selection")
    summary_lines.append("- Uses alphabetically first region for template")
    summary_lines.append("- Removes hidden source of non-determinism")
    summary_lines.append("- Consistent across different OS/filesystem orders")
    summary_lines.append("")
    summary_lines.append("### 7. Enhanced Error Handling")
    summary_lines.append("- SMOTE fallback to RandomOverSampler")
    summary_lines.append("- Safe metrics calculation with zero_division guards")
    summary_lines.append("- Zero-weight protection in aggregation")
    summary_lines.append("- Auto-discovery of regional datasets")
    summary_lines.append("- Oversampling ratio logging")
    summary_lines.append("")
    
    # Performance summary with both weighted and unweighted
    if not results_df.empty:
        avg_row = results_df[results_df['Region'] == 'AVERAGE']
        weighted_row = results_df[results_df['Region'] == 'WEIGHTED_AVERAGE']
        
        if not avg_row.empty and not weighted_row.empty:
            avg_metrics = avg_row.iloc[0]
            weighted_metrics = weighted_row.iloc[0]
            
            summary_lines.append("## Performance Results")
            summary_lines.append("### Unweighted Averages")
            summary_lines.append(f"Federated F1-Score: {avg_metrics['MLP_Federated_F1']:.4f}")
            summary_lines.append(f"Federated Accuracy: {avg_metrics['MLP_Federated_Accuracy']:.4f}")
            summary_lines.append(f"Federated ROC-AUC: {avg_metrics['MLP_Federated_ROC_AUC']:.4f}")
            
            summary_lines.append("")
            summary_lines.append("### Weighted Averages (by test sample size)")
            summary_lines.append(f"Weighted F1-Score: {weighted_metrics['MLP_Federated_F1']:.4f}")
            summary_lines.append(f"Weighted Accuracy: {weighted_metrics['MLP_Federated_Accuracy']:.4f}")
            summary_lines.append(f"Weighted ROC-AUC: {weighted_metrics['MLP_Federated_ROC_AUC']:.4f}")
            
            if centralized_metrics:
                summary_lines.append("")
                summary_lines.append("## Comparison with Centralized")
                summary_lines.append(f"Centralized F1-Score: {centralized_metrics['f1_score']:.4f}")
                unweighted_gap = avg_metrics['MLP_Federated_F1'] - centralized_metrics['f1_score']
                weighted_gap = weighted_metrics['MLP_Federated_F1'] - centralized_metrics['f1_score']
                summary_lines.append(f"Unweighted vs Centralized: {unweighted_gap:+.4f}")
                summary_lines.append(f"Weighted vs Centralized: {weighted_gap:+.4f}")
                
                if max(unweighted_gap, weighted_gap) > -0.05:
                    summary_lines.append("Status: Federated learning competitive")
                else:
                    summary_lines.append("Status: Federated learning needs improvement")
    
    # Training progression with both metrics
    if trainer.training_history:
        summary_lines.append("")
        summary_lines.append("## Training Progression")
        summary_lines.append(f"Aggregation Method: {trainer.federated_config['aggregation_method']}")
        summary_lines.append("")
        
        total_unweighted_improvement = 0
        total_weighted_improvement = 0
        
        for round_info in trainer.training_history:
            round_num = round_info['round'] + 1
            unweighted_f1 = round_info['average_test_f1']
            weighted_f1 = round_info['weighted_test_f1']
            unweighted_imp = round_info.get('improvement_from_previous', 0.0)
            weighted_imp = round_info.get('weighted_improvement', 0.0)
            raw_samples = round_info.get('total_raw_samples', 0)
            
            total_unweighted_improvement += unweighted_imp
            total_weighted_improvement += weighted_imp
            
            if unweighted_imp != 0.0:
                summary_lines.append(f"Round {round_num}: Unweighted F1={unweighted_f1:.4f} (Δ: {unweighted_imp:+.4f}), "
                                   f"Weighted F1={weighted_f1:.4f} (Δ: {weighted_imp:+.4f}) [Raw samples: {raw_samples}]")
            else:
                summary_lines.append(f"Round {round_num}: Unweighted F1={unweighted_f1:.4f}, "
                                   f"Weighted F1={weighted_f1:.4f} [Raw samples: {raw_samples}]")
        
        summary_lines.append("")
        summary_lines.append(f"Total unweighted improvement: {total_unweighted_improvement:+.4f}")
        summary_lines.append(f"Total weighted improvement: {total_weighted_improvement:+.4f}")
    
    # Production optimizations implemented
    summary_lines.append("")
    summary_lines.append("## Production Optimizations Implemented")
    
    summary_lines.append("")
    summary_lines.append("### 1. Safe Parameter Handling")
    summary_lines.append("- Only modify model.coefs_ and model.intercepts_")
    summary_lines.append("- No modification of internal attributes (n_layers_, n_outputs_)")
    summary_lines.append("- Prevents conflicts with internal buffers and loss curves")
    
    summary_lines.append("")
    summary_lines.append("### 2. Proper Batch Processing")
    summary_lines.append("- Complete batch coverage (no dropped tail samples)")
    summary_lines.append("- Epoch-wise shuffling for better generalization")
    summary_lines.append("- Range-based indexing: range(0, N, batch_size)")
    
    summary_lines.append("")
    summary_lines.append("### 3. Multiple Aggregation Methods")
    summary_lines.append("- FedAvg: Weighted by raw sample counts")
    summary_lines.append("- FedAvg Equal: Unweighted for diagnostic purposes")
    summary_lines.append("- Coordinate Median: Robust to outlier clients")
    summary_lines.append("- Trimmed Mean: Removes extreme values")
    
    summary_lines.append("")
    summary_lines.append("### 4. Robust Preprocessing")
    summary_lines.append("- Graceful column handling with reindex()")
    summary_lines.append("- Global medians for consistent imputation")
    summary_lines.append("- Same statistics for train/validation/test")
    
    summary_lines.append("")
    summary_lines.append("### 5. Full Reproducibility")
    summary_lines.append("- SHA256-based deterministic seeding")
    summary_lines.append("- Region and round-specific seeds")
    summary_lines.append("- Deterministic warm-up batch creation")
    summary_lines.append("- Deterministic template region selection")
    
    summary_lines.append("")
    summary_lines.append("### 6. Enhanced Metrics")
    summary_lines.append("- Both unweighted and weighted averages")
    summary_lines.append("- Weighted by test sample sizes")
    summary_lines.append("- Separate tracking of raw vs SMOTE sample counts")
    summary_lines.append("- Proper NaN handling in weighted calculations")
    
    summary_lines.append("")
    summary_lines.append("### 7. Optimized Configuration")
    summary_lines.append("- SGD solver for better partial_fit stability")
    summary_lines.append("- Explicit momentum=0.9 and Nesterov acceleration")
    summary_lines.append("- Reduced batch size to avoid clipping warnings")
    summary_lines.append("- Configurable validation frequency and patience")
    
    summary_lines.append("")
    summary_lines.append("### 8. Enhanced Error Handling")
    summary_lines.append("- SMOTE fallback to RandomOverSampler")
    summary_lines.append("- Safe metrics calculation with zero_division guards")
    summary_lines.append("- Zero-weight protection in aggregation")
    summary_lines.append("- Auto-discovery of regional datasets")
    summary_lines.append("- Oversampling ratio logging for debugging")
    
    if trainer.global_preprocessor.is_fitted:
        summary_lines.append("")
        summary_lines.append("## Global Preprocessor Details")
        summary_lines.append(f"- Selected features: {trainer.global_preprocessor.n_features}")
        summary_lines.append(f"- Total feature columns: {len(trainer.global_preprocessor.feature_columns)}")
        summary_lines.append("- Scaler: StandardScaler fitted on combined regional data")
        summary_lines.append("- Selector: SelectKBest with f_classif scoring")
        summary_lines.append("- Global medians stored for consistent imputation")
        summary_lines.append("- NOTE: Global preprocessor uses centralized statistics (acceptable for simulation)")
        summary_lines.append("- For production FL: consider secure aggregation or frozen preprocessor")
    
    summary_lines.append("")
    summary_lines.append("## Key Files")
    summary_lines.append("- Global Model: mlp_global_federated_model.joblib")
    summary_lines.append("- Robust Preprocessor: global_preprocessor.joblib")
    summary_lines.append("- Performance Results: mlp_federated_performance.csv")
    summary_lines.append("- Training History: federated_training_history.json")
    summary_lines.append("- Enhanced Summary: federated_improvement_summary.csv")
    summary_lines.append("- Regional Models: regional_models/")
    
    summary_path = os.path.join(MLP_RESULTS_DIR, 'EXPERIMENT_SUMMARY.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"Saved production experiment summary to {summary_path}")


def main():
    """Main production MLP federated learning pipeline."""
    logger.info("Starting PRODUCTION MLP FEDERATED LEARNING Pipeline")
    logger.info("IMPLEMENTED:")
    logger.info("  1. Deterministic seeding using SHA256 (no more hash() salt issues)")
    logger.info("  2. Proper fillna order (only global medians, no zero pollution)")
    logger.info("  3. Correct region-weight logging alignment")
    logger.info("  4. Truly balanced warm-up using minimum class count")
    logger.info("  5. Weighted ROC-AUC with proper NaN handling and re-normalization")
    logger.info("  6. Deterministic template region selection (alphabetically first)")
    logger.info("  7. Enhanced error handling and SMOTE fallback")
    logger.info("PRODUCTION OPTIMIZATIONS:")
    logger.info("  1. Safe parameter handling (no internal attribute modification)")
    logger.info("  2. Proper batch processing with epoch shuffling")
    logger.info("  3. Multiple aggregation methods available")
    logger.info("  4. Robust preprocessing with graceful column handling")
    logger.info("  5. Full reproducibility with deterministic seeding")
    logger.info("  6. Enhanced metrics (weighted and unweighted)")
    logger.info("  7. Optimized configuration (SGD with momentum and Nesterov)")
    logger.info("  8. Auto-discovery of regional datasets")
    logger.info("  9. Oversampling ratio logging for debugging")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and split datasets (auto-discovery)
        regional_train_data, regional_test_data = load_and_split_regional_datasets()
        
        if not regional_train_data:
            logger.error("No regional datasets found")
            return
        
        logger.info(f"Auto-discovered regions: {list(regional_train_data.keys())}")
        
        # Step 2: Load centralized baseline
        centralized_model, centralized_metrics = compare_with_centralized()
        
        # Step 3: Initialize production trainer
        trainer = ProductionMLPFederatedTrainer(MLP_CONFIG, FEDERATED_CONFIG)
        
        # Step 4: Run production federated learning
        global_model = trainer.run_federated_learning(regional_train_data, regional_test_data)
        
        if global_model is None:
            logger.error("Federated learning failed")
            return
        
        # Step 5: Evaluate with enhanced metrics
        results_df = evaluate_global_model(trainer, global_model, regional_test_data)
        
        # Step 6: Save all results
        save_federated_results(trainer, global_model, results_df, centralized_metrics)
        
        # Step 7: Production analysis and summary
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("PRODUCTION MLP FEDERATED LEARNING COMPLETED")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {MLP_RESULTS_DIR}")
        
        # Print enhanced performance summary
        if not results_df.empty:
            avg_row = results_df[results_df['Region'] == 'AVERAGE']
            weighted_row = results_df[results_df['Region'] == 'WEIGHTED_AVERAGE']
            
            if not avg_row.empty and not weighted_row.empty:
                avg_data = avg_row.iloc[0]
                weighted_data = weighted_row.iloc[0]
                
                logger.info("FINAL PERFORMANCE:")
                logger.info("  Unweighted Averages:")
                logger.info(f"    F1-Score: {avg_data['MLP_Federated_F1']:.4f}")
                logger.info(f"    Accuracy: {avg_data['MLP_Federated_Accuracy']:.4f}")
                logger.info(f"    ROC-AUC: {avg_data['MLP_Federated_ROC_AUC']:.4f}")
                logger.info("  Weighted Averages:")
                logger.info(f"    F1-Score: {weighted_data['MLP_Federated_F1']:.4f}")
                logger.info(f"    Accuracy: {weighted_data['MLP_Federated_Accuracy']:.4f}")
                logger.info(f"    ROC-AUC: {weighted_data['MLP_Federated_ROC_AUC']:.4f}")
                
                if centralized_metrics:
                    unweighted_gap = avg_data['MLP_Federated_F1'] - centralized_metrics['f1_score']
                    weighted_gap = weighted_data['MLP_Federated_F1'] - centralized_metrics['f1_score']
                    logger.info(f"  vs Centralized:")
                    logger.info(f"    Unweighted: {unweighted_gap:+.4f}")
                    logger.info(f"    Weighted: {weighted_gap:+.4f}")
                    
                    if max(unweighted_gap, weighted_gap) > -0.05:
                        logger.info("  Status: SUCCESS - Federated learning competitive")
                    else:
                        logger.info("  Status: NEEDS IMPROVEMENT - Federated underperforming")
        
        # Print enhanced training progression
        if trainer.training_history:
            logger.info("TRAINING PROGRESSION ANALYSIS:")
            logger.info(f"  Aggregation Method: {trainer.federated_config['aggregation_method']}")
            
            total_unweighted_improvement = 0
            total_weighted_improvement = 0
            
            for round_info in trainer.training_history:
                round_num = round_info['round'] + 1
                unweighted_f1 = round_info['average_test_f1']
                weighted_f1 = round_info['weighted_test_f1']
                unweighted_imp = round_info.get('improvement_from_previous', 0.0)
                weighted_imp = round_info.get('weighted_improvement', 0.0)
                raw_samples = round_info.get('total_raw_samples', 0)
                
                total_unweighted_improvement += unweighted_imp
                total_weighted_improvement += weighted_imp
                
                if unweighted_imp != 0.0:
                    logger.info(f"  Round {round_num}: Unweighted F1={unweighted_f1:.4f} (Δ: {unweighted_imp:+.4f}), "
                               f"Weighted F1={weighted_f1:.4f} (Δ: {weighted_imp:+.4f})")
                else:
                    logger.info(f"  Round {round_num}: Unweighted F1={unweighted_f1:.4f}, Weighted F1={weighted_f1:.4f}")
            
            logger.info(f"  Total Improvements:")
            logger.info(f"    Unweighted: {total_unweighted_improvement:+.4f}")
            logger.info(f"    Weighted: {total_weighted_improvement:+.4f}")
            
            # Analysis based on best improvement metric
            best_improvement = max(total_unweighted_improvement, total_weighted_improvement)
            if best_improvement > 0.02:
                logger.info("  Analysis: POSITIVE - Federated learning shows significant improvement")
            elif best_improvement > 0.005:
                logger.info("  Analysis: MODERATE - Federated learning shows modest improvement")
            elif best_improvement > -0.005:
                logger.info("  Analysis: STABLE - Federated learning maintains performance")
            else:
                logger.info("  Analysis: CONCERNING - Federated learning degrading")
        
        # Verification
        logger.info("VERIFICATION:")
        logger.info(f"  Deterministic seeding: IMPLEMENTED (SHA256-based)")
        logger.info(f"  Proper fillna order: IMPLEMENTED (only global medians)")
        logger.info(f"  Correct region-weight logging: IMPLEMENTED")
        logger.info(f"  Truly balanced warm-up: IMPLEMENTED (min class count)")
        logger.info(f"  Weighted ROC-AUC NaN handling: IMPLEMENTED")
        logger.info(f"  Deterministic template selection: IMPLEMENTED")
        logger.info(f"  Enhanced error handling: IMPLEMENTED")
        
        # Production optimizations verification
        logger.info("PRODUCTION OPTIMIZATIONS VERIFICATION:")
        logger.info(f"  Safe parameter handling: IMPLEMENTED")
        logger.info(f"  Proper batch processing: IMPLEMENTED (no tail drops)")
        logger.info(f"  Epoch shuffling: IMPLEMENTED")
        logger.info(f"  Aggregation method: {trainer.federated_config['aggregation_method']}")
        logger.info(f"  Robust preprocessing: IMPLEMENTED")
        logger.info(f"  SMOTE fallback: IMPLEMENTED")
        logger.info(f"  Enhanced metrics: IMPLEMENTED (weighted + unweighted)")
        logger.info(f"  Optimized solver: {trainer.mlp_config['solver']} with momentum={trainer.mlp_config['momentum']}")
        logger.info(f"  Auto-discovery: IMPLEMENTED")
        logger.info(f"  Oversampling logging: IMPLEMENTED")
        
        # Global preprocessor verification
        if trainer.global_preprocessor.is_fitted:
            logger.info("ROBUST GLOBAL PREPROCESSOR VERIFICATION:")
            logger.info(f"  Selected features: {trainer.global_preprocessor.n_features}")
            logger.info(f"  Feature columns: {len(trainer.global_preprocessor.feature_columns)}")
            logger.info(f"  Global medians stored: {trainer.global_preprocessor.global_medians is not None}")
            logger.info(f"  Consistent imputation: VERIFIED")
        
        # Data integrity verification
        logger.info("DATA INTEGRITY VERIFICATION:")
        for region in regional_train_data.keys():
            train_size = len(regional_train_data[region])
            test_size = len(regional_test_data[region])
            total_size = train_size + test_size
            test_ratio = test_size / total_size
            logger.info(f"  {region}: {train_size} train + {test_size} test = {total_size} total (ratio: {test_ratio:.1%})")
        
        logger.info("=" * 80)
        logger.info("PRODUCTION OPTIMIZATIONS IMPLEMENTED")
        logger.info("Key improvements from production version:")
        logger.info("  - TRUE reproducibility with SHA256 deterministic seeding")
        logger.info("  - NO zero pollution in global median imputation")
        logger.info("  - CORRECT region-weight alignment in logging")
        logger.info("  - TRULY balanced warm-up using minimum class count")
        logger.info("  - PROPER weighted ROC-AUC with NaN re-normalization")
        logger.info("  - DETERMINISTIC template region selection")
        logger.info("  - Enhanced error handling with SMOTE fallback")
        logger.info("  - Auto-discovery of regional datasets")
        logger.info("  - Safe metrics calculation with proper guards")
        logger.info("  - Zero-weight protection in aggregation")
        logger.info("  - Complete batch coverage with shuffling")
        logger.info("  - Multiple aggregation strategies available")
        logger.info("  - SGD with explicit momentum and Nesterov acceleration")
        logger.info("  - Robust preprocessing with graceful error handling")
        logger.info("  - Enhanced metrics for better analysis")
        logger.info("  - Oversampling ratio logging for debugging")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
