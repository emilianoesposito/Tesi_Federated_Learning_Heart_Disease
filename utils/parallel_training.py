# -*- coding: utf-8 -*-
import joblib
import json
import os
import time
import psutil
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE
import optuna
import xgboost as xgb
import lightgbm as lgb


class SystemResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.cpu = []
        self.memory = []

    def start(self):
        from threading import Thread
        self.monitoring = True
        def loop():
            while self.monitoring:
                self.cpu.append(psutil.cpu_percent())
                self.memory.append(psutil.virtual_memory().percent)
                time.sleep(1)
        Thread(target=loop, daemon=True).start()

    def stop(self):
        self.monitoring = False

    def stats(self):
        return {
            'avg_cpu': np.mean(self.cpu) if self.cpu else 0,
            'avg_mem': np.mean(self.memory) if self.memory else 0
        }


class ParallelHyperparameterOptimizer:
    def __init__(self, random_state=42, n_trials=2):
        self.random_state = random_state
        self.n_trials = n_trials

    def optimize_random_forest(self, X, y):
        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=2
            )
            scores = []
            skf = StratifiedKFold(n_splits=3)
            for train_idx, val_idx in skf.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                scores.append(f1_score(y[val_idx], preds))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
        return study

    def optimize_xgboost(self, X, y):
        def objective(trial):
            model = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=2,
                random_state=self.random_state
            )
            scores = []
            skf = StratifiedKFold(n_splits=3)
            for train_idx, val_idx in skf.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                scores.append(f1_score(y[val_idx], preds))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
        return study

    def optimize_lightgbm(self, X, y):
        def objective(trial):
            model = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=2
            )
            scores = []
            skf = StratifiedKFold(n_splits=3)
            for train_idx, val_idx in skf.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                scores.append(f1_score(y[val_idx], preds))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
        return study


class ParallelModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.monitor = SystemResourceMonitor()
        self.optimizer = ParallelHyperparameterOptimizer(random_state=random_state, n_trials=50)

    def parallel_hyperparameter_optimization(self, X, y):
        print("🎯 Starting hyperparameter optimization...")
        self.monitor.start()
        best_params = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(getattr(self.optimizer, f"optimize_{name}"), X, y): name
                for name in ["random_forest", "xgboost", "lightgbm"]
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    study = future.result()
                    best_params[name] = study.best_params if study else {}
                    print(f"✅ {name.upper()} optimized")
                except Exception as e:
                    print(f"❌ {name.upper()} optimization failed: {e}")
                    best_params[name] = {}

        self.monitor.stop()
        stats = self.monitor.stats()
        print(f"🧠 Avg CPU: {stats['avg_cpu']:.1f}%, Avg RAM: {stats['avg_mem']:.1f}%")
        return best_params

    def create_optimized_models(self, best_params):
        models = []

        models.append({
            'name': 'RandomForest_Optimized',
            'class': RandomForestClassifier,
            'params': {**best_params.get('random_forest', {}),
                       'class_weight': 'balanced', 'n_jobs': 4, 'random_state': self.random_state}
        })

        models.append({
            'name': 'XGBoost_Optimized',
            'class': xgb.XGBClassifier,
            'params': {**best_params.get('xgboost', {}),
                       'use_label_encoder': False, 'eval_metric': 'logloss',
                       'n_jobs': 4, 'random_state': self.random_state}
        })

        models.append({
            'name': 'LightGBM_Optimized',
            'class': lgb.LGBMClassifier,
            'params': {**best_params.get('lightgbm', {}),
                       'class_weight': 'balanced', 'n_jobs': 4, 'random_state': self.random_state}
        })

        models.append({
            'name': 'ExtraTrees',
            'class': ExtraTreesClassifier,
            'params': {'n_estimators': 200, 'max_depth': 10,
                       'class_weight': 'balanced', 'n_jobs': 4, 'random_state': self.random_state}
        })

        models.append({
            'name': 'GradientBoosting',
            'class': GradientBoostingClassifier,
            'params': {'n_estimators': 200, 'max_depth': 5,
                       'learning_rate': 0.1, 'random_state': self.random_state}
        })

        models.append({
            'name': 'HistGradientBoosting',
            'class': HistGradientBoostingClassifier,
            'params': {'max_iter': 200, 'random_state': self.random_state}
        })

        models.append({
            'name': 'MLP_Optimized',
            'class': MLPClassifier,
            'params': {'hidden_layer_sizes': (128, 64), 'activation': 'relu',
                       'solver': 'adam', 'alpha': 0.01, 'batch_size': 512,
                       'learning_rate': 'adaptive', 'max_iter': 300,
                       'early_stopping': True, 'validation_fraction': 0.1,
                       'n_iter_no_change': 10, 'random_state': self.random_state}
        })

        return models

    def train_model(self, config, X_train, y_train, X_test, y_test):
        name = config['name']
        model_class = config['class']
        params = config['params']
        print(f"🔄 Training {name}...")
        start = time.time()

        try:
            model = model_class(**params)
            model.fit(X_train, y_train)

            calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated.fit(X_train, y_train)

            preds = calibrated.predict(X_test)
            probs = calibrated.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, preds),
                'precision': precision_score(y_test, preds),
                'recall': recall_score(y_test, preds),
                'f1_score': f1_score(y_test, preds),
                'roc_auc': roc_auc_score(y_test, probs)
            }

            print(f"✅ {name} trained in {time.time() - start:.1f}s - F1: {metrics['f1_score']:.4f}")
            return {
                'name': name,
                'model': calibrated,
                'metrics': metrics,
                'status': 'success',
                'training_time': time.time() - start
            }
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return {'name': name, 'status': 'failed', 'error': str(e)}

    def parallel_model_training(self, model_configs, X_train, y_train, X_test, y_test):
        print(f"🚀 Training {len(model_configs)} models in parallel...")
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(model_configs), 6)) as executor:
            futures = {
                executor.submit(self.train_model, cfg, X_train, y_train, X_test, y_test): cfg['name']
                for cfg in model_configs
            }

            for future in as_completed(futures):
                res = future.result()
                if res['status'] == 'success':
                    results[res['name']] = res
                else:
                    print(f"⚠️ {res['name']} failed: {res.get('error', 'Unknown error')}")

        return results

    def create_ensemble_model(self, results, X_train, y_train):
        print("🧩 Building ensemble model...")

        successful = [(name, results[name]['model']) for name in results if results[name]['status'] == 'success']

        if len(successful) < 2:
            print("⚠️ Not enough models for ensemble. Returning None.")
            return None

        ensemble_model = VotingClassifier(
            estimators=successful,
            voting='soft',
            n_jobs=4
        )

        calibrated_ensemble = CalibratedClassifierCV(ensemble_model, method='isotonic', cv=3)
        calibrated_ensemble.fit(X_train, y_train)

        print("✅ Ensemble model trained.")
        return calibrated_ensemble
  

    def save_models(self, results, ensemble_model, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)

        print("💾 Saving individual models...")
        for name, info in results.items():
            if info['status'] == 'success':
                model_path = os.path.join(save_dir, f"{name}.joblib")
                joblib.dump(info['model'], model_path)

        if ensemble_model:
            print("💾 Saving ensemble model...")
            joblib.dump(ensemble_model, os.path.join(save_dir, "ensemble_model.joblib"))

        print("📈 Saving metrics summary...")
        metrics_data = {
            name: info['metrics'] for name, info in results.items() if info['status'] == 'success'
        }
        metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
        metrics_df.to_csv(os.path.join(save_dir, "metrics_summary.csv"))

        print("✅ All models and metrics saved to 'results/'")



def prepare_data_for_training(df_train, test_size=0.2, random_state=42):
    print("📊 Preparing data for training...")
    y = df_train["outcome"]
    X = df_train.drop(columns=["outcome"]).fillna(df_train.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_classif, k=min(50, X_train_scaled.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    smote = SMOTE(random_state=random_state)
    X_train_final, y_train_final = smote.fit_resample(X_train_sel, y_train)

    print(f"✅ Final training samples: {X_train_final.shape[0]}")
    return {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "X_test": X_test_sel,
        "y_test": y_test,
        "scaler": scaler,
        "selector": selector
    }
