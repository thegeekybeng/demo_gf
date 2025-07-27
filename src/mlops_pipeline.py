"""
MLOps Pipeline for Global Foundries Wafer Manufacturing
Automated model training, deployment, and monitoring system
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
from dataclasses import dataclass
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    name: str
    version: str
    target_metric: str
    threshold: float
    retrain_frequency_hours: int
    features: List[str]
    hyperparameters: Dict

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    rmse: float
    mae: float
    r2_score: float
    feature_importance: Dict[str, float]
    training_time: float
    prediction_latency: float

class DataDriftDetector:
    """Detect data drift in incoming wafer data"""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._calculate_stats(reference_data)
    
    def _calculate_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical properties of the data"""
        return {
            'mean': data.mean().to_dict(),
            'std': data.std().to_dict(),
            'min': data.min().to_dict(),
            'max': data.max().to_dict()
        }
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, bool]:
        """Detect if new data has drifted from reference"""
        new_stats = self._calculate_stats(new_data)
        drift_detected = {}
        
        for column in new_data.columns:
            if column in self.reference_stats['mean']:
                # Simple drift detection using mean difference
                mean_diff = abs(new_stats['mean'][column] - self.reference_stats['mean'][column])
                std_threshold = self.reference_stats['std'][column] * self.threshold
                drift_detected[column] = mean_diff > std_threshold
        
        return drift_detected

class YieldPredictionModel:
    """ML model for predicting wafer yield"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.metrics = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for yield prediction"""
        features = data.copy()
        
        # Temperature-based features
        features['temp_deviation'] = abs(features['temperature'] - 1050)
        features['temp_squared'] = features['temperature'] ** 2
        
        # Pressure features
        features['pressure_deviation'] = abs(features['pressure'] - 10.0)
        
        # Interaction features
        features['temp_pressure_interaction'] = features['temperature'] * features['pressure']
        features['time_temp_interaction'] = features['etch_time'] * features['temperature']
        
        # Rolling statistics (if time series data)
        if 'timestamp' in features.columns:
            features = features.sort_values('timestamp')
            features['temp_rolling_mean'] = features['temperature'].rolling(window=10).mean()
            features['pressure_rolling_std'] = features['pressure'].rolling(window=10).std()
        
        return features[self.config.features]
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ModelMetrics:
        """Train the yield prediction model"""
        start_time = datetime.now()
        
        # Prepare features
        X_train_features = self.prepare_features(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Initialize model with hyperparameters
        self.model = RandomForestRegressor(**self.config.hyperparameters)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.config.features,
            self.model.feature_importances_
        ))
        
        # Calculate metrics
        y_pred = self.model.predict(X_train_scaled)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.metrics = ModelMetrics(
            accuracy=r2_score(y_train, y_pred),
            rmse=np.sqrt(mean_squared_error(y_train, y_pred)),
            mae=mean_absolute_error(y_train, y_pred),
            r2_score=r2_score(y_train, y_pred),
            feature_importance=self.feature_importance,
            training_time=training_time,
            prediction_latency=0.0  # Will be measured during inference
        )
        
        logger.info("Model trained. R² Score: %.4f", self.metrics.r2_score)
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Make predictions and measure latency"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        start_time = datetime.now()
        
        X_features = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)
        
        latency = (datetime.now() - start_time).total_seconds()
        return predictions, latency
    
    def save_model(self, path: str):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, path)
        logger.info("Model saved to %s", path)
    
    def load_model(self, path: str):
        """Load model and scaler"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.metrics = model_data['metrics']
        self.feature_importance = model_data['feature_importance']
        logger.info("Model loaded from %s", path)

class DefectClassifier:
    """ML model for classifying defect patterns"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = RandomForestRegressor(**config.hyperparameters)
        self.scaler = StandardScaler()
    
    def extract_wafer_features(self, wafer_maps: np.ndarray) -> pd.DataFrame:
        """Extract features from wafer map images"""
        features = []
        
        for wafer in wafer_maps:
            feature_dict = {
                'total_defects': np.sum(wafer > 0.5),
                'defect_density': np.mean(wafer > 0.5),
                'center_defects': np.sum(wafer[20:32, 20:32] > 0.5),
                'edge_defects': self._count_edge_defects(wafer),
                'symmetry_score': self._calculate_symmetry(wafer),
                'cluster_count': self._count_clusters(wafer),
                'max_defect_size': self._max_defect_size(wafer),
                'defect_spread': self._defect_spread(wafer)
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _count_edge_defects(self, wafer: np.ndarray) -> int:
        """Count defects near edges"""
        edge_mask = np.zeros_like(wafer)
        edge_mask[:5, :] = 1
        edge_mask[-5:, :] = 1
        edge_mask[:, :5] = 1
        edge_mask[:, -5:] = 1
        return np.sum((wafer > 0.5) & (edge_mask == 1))
    
    def _calculate_symmetry(self, wafer: np.ndarray) -> float:
        """Calculate symmetry of defect pattern"""
        left_half = wafer[:, :wafer.shape[1]//2]
        right_half = np.fliplr(wafer[:, wafer.shape[1]//2:])
        
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        return float(1 - np.mean(np.abs(left_half - right_half)))
    
    def _count_clusters(self, wafer: np.ndarray) -> int:
        """Count number of defect clusters"""
        # Simplified cluster counting
        defect_mask = wafer > 0.5
        return int(np.sum(defect_mask) // 10)  # Approximate cluster count
    
    def _max_defect_size(self, wafer: np.ndarray) -> int:
        """Find maximum contiguous defect size"""
        defect_mask = wafer > 0.5
        return np.max(defect_mask.sum(axis=1))
    
    def _defect_spread(self, wafer: np.ndarray) -> float:
        """Calculate how spread out defects are"""
        defect_positions = np.where(wafer > 0.5)
        if len(defect_positions[0]) == 0:
            return 0.0
        return float(np.std(defect_positions[0]) + np.std(defect_positions[1]))

class AnomalyDetector:
    """Detect anomalous process conditions"""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame):
        """Fit anomaly detection model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("Anomaly detector fitted")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (-1 = anomaly, 1 = normal)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class MLOpsMonitor:
    """Monitor model performance and trigger retraining"""
    
    def __init__(self, models: Dict[str, Any], drift_detector: DataDriftDetector):
        self.models = models
        self.drift_detector = drift_detector
        self.performance_history = []
        
    def log_prediction(self, model_name: str, prediction: float, actual: Optional[float] = None,
                      features: Optional[Dict] = None, timestamp: Optional[datetime] = None):
        """Log prediction for monitoring"""
        log_entry = {
            'model_name': model_name,
            'prediction': prediction,
            'actual': actual,
            'features': features,
            'timestamp': timestamp or datetime.now(),
            'error': None if actual is None else abs(prediction - actual)
        }
        self.performance_history.append(log_entry)
    
    def check_model_performance(self, model_name: str, window_hours: int = 24) -> Dict:
        """Check recent model performance"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_predictions = [
            entry for entry in self.performance_history
            if entry['model_name'] == model_name and 
               entry['timestamp'] > cutoff_time and
               entry['actual'] is not None
        ]
        
        if not recent_predictions:
            return {'status': 'insufficient_data', 'count': 0}
        
        errors = [entry['error'] for entry in recent_predictions]
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        return {
            'status': 'ok',
            'count': len(recent_predictions),
            'mae': mae,
            'rmse': rmse,
            'predictions': recent_predictions
        }
    
    def should_retrain(self, model_name: str, performance_threshold: float = 0.1) -> Dict:
        """Determine if model should be retrained"""
        performance = self.check_model_performance(model_name)
        
        reasons = []
        
        # Check performance degradation
        if performance['status'] == 'ok' and performance['mae'] > performance_threshold:
            reasons.append(f"Performance degraded: MAE = {performance['mae']:.4f}")
        
        # Check data drift
        if len(self.performance_history) > 100:
            recent_features = pd.DataFrame([
                entry['features'] for entry in self.performance_history[-100:]
                if entry['features'] is not None
            ])
            
            if not recent_features.empty:
                drift_results = self.drift_detector.detect_drift(recent_features)
                drifted_features = [feat for feat, is_drift in drift_results.items() if is_drift]
                
                if drifted_features:
                    reasons.append(f"Data drift detected in: {', '.join(drifted_features)}")
        
        return {
            'should_retrain': len(reasons) > 0,
            'reasons': reasons,
            'performance': performance
        }

class MLOpsPipeline:
    """Complete MLOps pipeline for wafer manufacturing"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.monitor = None
        self.drift_detector = None
        
        # Initialize MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def initialize_models(self, training_data: pd.DataFrame):
        """Initialize all models in the pipeline"""
        
        # Initialize yield prediction model
        yield_config = ModelConfig(**self.config['models']['yield_prediction'])
        self.models['yield_prediction'] = YieldPredictionModel(yield_config)
        
        # Initialize defect classifier
        defect_config = ModelConfig(**self.config['models']['defect_classification'])
        self.models['defect_classification'] = DefectClassifier(defect_config)
        
        # Initialize anomaly detector
        self.models['anomaly_detection'] = AnomalyDetector(
            contamination=self.config['models']['anomaly_detection']['contamination']
        )
        
        # Initialize drift detector
        self.drift_detector = DataDriftDetector(
            training_data[yield_config.features],
            threshold=self.config['drift_detection']['threshold']
        )
        
        # Initialize monitor
        self.monitor = MLOpsMonitor(self.models, self.drift_detector)
        
        logger.info("All models initialized")
    
    def train_pipeline(self, data: pd.DataFrame, target_column: str = 'yield_rate'):
        """Train all models in the pipeline"""
        
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"pipeline_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                return self._train_models(data, target_column)
        else:
            return self._train_models(data, target_column)
    
    def _train_models(self, data: pd.DataFrame, target_column: str):
        """Internal method to train models"""
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train yield prediction model
        yield_metrics = self.models['yield_prediction'].train(X_train, y_train)
        
        # Log yield model metrics
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'yield_r2_score': yield_metrics.r2_score,
                'yield_rmse': yield_metrics.rmse,
                'yield_mae': yield_metrics.mae,
                'yield_training_time': yield_metrics.training_time
            })
        
        # Train anomaly detector
        self.models['anomaly_detection'].fit(X_train)
        
        # Save models
        model_dir = f"models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models['yield_prediction'].save_model(f"{model_dir}/yield_model.pkl")
        
        # Log model artifacts
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(
                self.models['yield_prediction'].model,
                "yield_prediction_model"
            )
        
        logger.info("Pipeline training completed. Models saved to %s", model_dir)
        
        return {
            'yield_metrics': yield_metrics,
            'model_dir': model_dir
        }
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """Make predictions using the pipeline"""
        
        # Yield prediction
        yield_pred, yield_latency = self.models['yield_prediction'].predict(data)
        
        # Anomaly detection
        anomalies = self.models['anomaly_detection'].predict(data)
        
        # Log predictions for monitoring
        if self.monitor is not None:
            for i, yield_val in enumerate(yield_pred):
                self.monitor.log_prediction(
                    'yield_prediction',
                    yield_val,
                    features=data.iloc[i].to_dict()
                )
        
        return {
            'yield_predictions': yield_pred.tolist(),
            'anomalies': anomalies.tolist(),
            'latency': {
                'yield_prediction': yield_latency
            },
            'model_performance': self.monitor.check_model_performance('yield_prediction') if self.monitor else None
        }
    
    def check_and_retrain(self) -> Dict:
        """Check if retraining is needed and execute if necessary"""
        
        retrain_decisions = {}
        
        if self.monitor is None:
            return {'error': 'Monitor not initialized'}
        
        for model_name in ['yield_prediction']:
            decision = self.monitor.should_retrain(model_name)
            retrain_decisions[model_name] = decision
            
            if decision['should_retrain']:
                logger.info("Retraining %s: %s", model_name, decision['reasons'])
                # Here you would trigger the retraining process
                # This could be a separate service or pipeline
        
        return retrain_decisions
    
    def get_pipeline_status(self) -> Dict:
        """Get overall pipeline health status"""
        
        status = {
            'models': {},
            'drift_status': {},
            'last_updated': datetime.now().isoformat(),
            'pipeline_health': 'healthy'
        }
        
        if self.monitor is None:
            status['pipeline_health'] = 'monitor_not_initialized'
            return status
        
        # Check each model
        for model_name in self.models:
            if model_name in ['yield_prediction']:
                perf = self.monitor.check_model_performance(model_name)
                status['models'][model_name] = perf
        
        # Check for any issues
        issues = []
        for model_name, model_status in status['models'].items():
            if model_status.get('mae', 0) > 0.1:
                issues.append(f"{model_name} performance degraded")
        
        if issues:
            status['pipeline_health'] = 'degraded'
            status['issues'] = issues
        
        return status

def load_pipeline_config() -> Dict:
    """Load MLOps pipeline configuration"""
    return {
        'mlflow': {
            'tracking_uri': 'sqlite:///mlflow.db',
            'experiment_name': 'wafer_manufacturing_optimization'
        },
        'models': {
            'yield_prediction': {
                'name': 'yield_predictor',
                'version': '1.0',
                'target_metric': 'r2_score',
                'threshold': 0.85,
                'retrain_frequency_hours': 24,
                'features': ['temperature', 'pressure', 'etch_time', 'deposition_rate', 'chamber_flow'],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'defect_classification': {
                'name': 'defect_classifier',
                'version': '1.0',
                'target_metric': 'f1_score',
                'threshold': 0.9,
                'retrain_frequency_hours': 48,
                'features': ['total_defects', 'defect_density', 'center_defects', 'edge_defects'],
                'hyperparameters': {
                    'n_estimators': 150,
                    'max_depth': 15,
                    'min_samples_split': 3,
                    'random_state': 42
                }
            },
            'anomaly_detection': {
                'contamination': 0.1
            }
        },
        'drift_detection': {
            'threshold': 0.1
        }
    }
