"""
Model Training Pipeline
Trains SVD, NMF models with hyperparameter optimization and MLflow tracking
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
import structlog
import yaml
from delta import DeltaTable

from ..utils.metrics import RecommendationMetrics
from ..streaming.kafka_producer import KafkaProducer

logger = structlog.get_logger()

class ModelTrainer:
    """Advanced model training with hyperparameter optimization"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Spark for data processing
        self.spark = SparkSession.builder \
            .appName("ModelTraining") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        # MLflow setup
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.metrics_calculator = RecommendationMetrics()
        
        # Target metrics from project specification
        self.target_metrics = {
            'rmse': 0.84,
            'ndcg_10': 0.78,
            'map_10': 0.73,
            'hit_rate_20': 0.91,
            'user_coverage': 0.942,
            'catalog_coverage': 0.785,
            'r2_score': 0.89
        }
    
    def load_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and prepare training data from Delta Lake"""
        try:
            # Load interactions from Delta Lake
            interactions_df = self.spark.read.format("delta").load("/delta/interactions")
            interactions_pd = interactions_df.toPandas()
            
            logger.info(f"Loaded {len(interactions_pd)} interactions")
            
            # Create user-item matrix
            user_item_matrix = interactions_pd.pivot(
                index='user_id',
                columns='item_id', 
                values='rating'
            ).fillna(0)
            
            logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
            
            return interactions_pd, user_item_matrix.values
            
        except Exception as e:
            logger.warning(f"Could not load data from Delta Lake: {e}")
            # Generate synthetic data for demonstration
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic data for demonstration"""
        logger.info("Generating synthetic training data...")
        
        np.random.seed(42)
        n_users, n_items = 10000, 2000
        density = 0.05  # 5% of possible interactions
        
        n_interactions = int(n_users * n_items * density)
        
        # Generate interactions
        user_ids = np.random.randint(0, n_users, n_interactions)
        item_ids = np.random.randint(0, n_items, n_interactions)
        ratings = np.random.normal(3.5, 1.0, n_interactions)
        ratings = np.clip(ratings, 1, 5)
        
        interactions_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': time.time()
        })
        
        # Remove duplicates and keep last rating
        interactions_df = interactions_df.groupby(['user_id', 'item_id']).last().reset_index()
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)
        
        logger.info(f"Generated {len(interactions_df)} unique interactions")
        logger.info(f"Matrix shape: {user_item_matrix.shape}")
        
        return interactions_df, user_item_matrix.values
    
    def train_svd_model(self, user_item_matrix: np.ndarray) -> Tuple[TruncatedSVD, Dict[str, float]]:
        """Train SVD model with hyperparameter optimization"""
        logger.info("Training SVD model...")
        
        with mlflow.start_run(run_name="svd_training") as run:
            # Hyperparameter grid
            param_grid = {
                'n_components': [50, 100, 150, 200],
                'n_iter': [10, 20, 50],
                'random_state': [42]
            }
            
            # Grid search for best parameters
            best_rmse = float('inf')
            best_model = None
            best_params = None
            
            for n_components in param_grid['n_components']:
                for n_iter in param_grid['n_iter']:
                    # Train model
                    svd = TruncatedSVD(
                        n_components=n_components,
                        n_iter=n_iter,
                        random_state=42
                    )
                    
                    # Split data for validation
                    train_matrix, val_matrix = self._split_matrix(user_item_matrix, test_size=0.2)
                    
                    # Fit model
                    svd.fit(train_matrix)
                    
                    # Predict and evaluate
                    reconstructed = svd.inverse_transform(svd.transform(val_matrix))
                    
                    # Calculate RMSE only on non-zero entries
                    mask = val_matrix > 0
                    if np.sum(mask) > 0:
                        rmse = np.sqrt(mean_squared_error(val_matrix[mask], reconstructed[mask]))
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_model = svd
                            best_params = {'n_components': n_components, 'n_iter': n_iter}
                        
                        logger.info(f"SVD params: {n_components}, {n_iter} - RMSE: {rmse:.4f}")
            
            # Retrain best model on full data
            final_svd = TruncatedSVD(**best_params, random_state=42)
            final_svd.fit(user_item_matrix)
            
            # Calculate final metrics
            metrics = self._calculate_svd_metrics(final_svd, user_item_matrix)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_svd, "svd_model")
            
            # Log variance explained
            variance_explained = np.sum(final_svd.explained_variance_ratio_)
            mlflow.log_metric("variance_explained", variance_explained)
            
            logger.info(f"Best SVD model - RMSE: {best_rmse:.4f}, Params: {best_params}")
            logger.info(f"Variance explained: {variance_explained:.4f}")
            
            return final_svd, metrics
    
    def train_nmf_model(self, user_item_matrix: np.ndarray) -> Tuple[NMF, Dict[str, float]]:
        """Train NMF model with hyperparameter optimization"""
        logger.info("Training NMF model...")
        
        # Ensure non-negative matrix
        user_item_matrix = np.maximum(0, user_item_matrix)
        
        with mlflow.start_run(run_name="nmf_training") as run:
            # Hyperparameter grid
            param_grid = {
                'n_components': [30, 50, 80, 100],
                'alpha': [0.0001, 0.001, 0.01],
                'l1_ratio': [0.0, 0.1, 0.5],
                'max_iter': [200, 500]
            }
            
            best_rmse = float('inf')
            best_model = None
            best_params = None
            
            for n_components in param_grid['n_components']:
                for alpha in param_grid['alpha']:
                    for l1_ratio in param_grid['l1_ratio']:
                        for max_iter in param_grid['max_iter']:
                            try:
                                # Train model
                                nmf = NMF(
                                    n_components=n_components,
                                    alpha=alpha,
                                    l1_ratio=l1_ratio,
                                    max_iter=max_iter,
                                    random_state=42,
                                    init='random'
                                )
                                
                                # Split data
                                train_matrix, val_matrix = self._split_matrix(user_item_matrix, test_size=0.2)
                                
                                # Fit model
                                W = nmf.fit_transform(train_matrix)
                                H = nmf.components_
                                
                                # Predict
                                reconstructed = np.dot(W, H)
                                
                                # Evaluate on validation set
                                val_W = nmf.transform(val_matrix)
                                val_reconstructed = np.dot(val_W, H)
                                
                                # Calculate RMSE
                                mask = val_matrix > 0
                                if np.sum(mask) > 0:
                                    rmse = np.sqrt(mean_squared_error(val_matrix[mask], val_reconstructed[mask]))
                                    
                                    if rmse < best_rmse:
                                        best_rmse = rmse
                                        best_model = nmf
                                        best_params = {
                                            'n_components': n_components,
                                            'alpha': alpha,
                                            'l1_ratio': l1_ratio,
                                            'max_iter': max_iter
                                        }
                                    
                                    logger.info(f"NMF params: {n_components}, {alpha}, {l1_ratio} - RMSE: {rmse:.4f}")
                                    
                            except Exception as e:
                                logger.warning(f"NMF training failed for params {n_components}, {alpha}: {e}")
                                continue
            
            # Retrain best model on full data
            final_nmf = NMF(**best_params, random_state=42, init='random')
            final_nmf.fit(user_item_matrix)
            
            # Calculate final metrics
            metrics = self._calculate_nmf_metrics(final_nmf, user_item_matrix)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_nmf, "nmf_model")
            
            logger.info(f"Best NMF model - RMSE: {best_rmse:.4f}, Params: {best_params}")
            
            return final_nmf, metrics
    
    def _split_matrix(self, matrix: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Split user-item matrix for training and validation"""
        train_matrix = matrix.copy()
        val_matrix = np.zeros_like(matrix)
        
        # For each user, move some ratings to validation set
        for i in range(matrix.shape[0]):
            user_ratings = np.nonzero(matrix[i, :])[0]
            if len(user_ratings) > 1:
                n_val = max(1, int(len(user_ratings) * test_size))
                val_indices = np.random.choice(user_ratings, n_val, replace=False)
                
                val_matrix[i, val_indices] = matrix[i, val_indices]
                train_matrix[i, val_indices] = 0
        
        return train_matrix, val_matrix
    
    def _calculate_svd_metrics(self, model: TruncatedSVD, matrix: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for SVD model"""
        try:
            # Reconstruct matrix
            reconstructed = model.inverse_transform(model.transform(matrix))
            
            # Calculate RMSE on non-zero entries
            mask = matrix > 0
            rmse = np.sqrt(mean_squared_error(matrix[mask], reconstructed[mask])) if np.sum(mask) > 0 else 0
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(matrix[mask], reconstructed[mask]) if np.sum(mask) > 0 else 0
            
            # Variance explained
            variance_explained = np.sum(model.explained_variance_ratio_)
            
            metrics = {
                'rmse': rmse,
                'r2_score': r2,
                'variance_explained': variance_explained,
                'n_components': model.n_components
            }
            
            # Add target achievement metrics
            metrics['rmse_target_achievement'] = self.target_metrics['rmse'] / rmse if rmse > 0 else 0
            metrics['r2_target_achievement'] = r2 / self.target_metrics['r2_score']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating SVD metrics: {e}")
            return {'rmse': float('inf'), 'r2_score': 0}
    
    def _calculate_nmf_metrics(self, model: NMF, matrix: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for NMF model"""
        try:
            # Reconstruct matrix
            W = model.transform(matrix)
            H = model.components_
            reconstructed = np.dot(W, H)
            
            # Calculate RMSE on non-zero entries
            mask = matrix > 0
            rmse = np.sqrt(mean_squared_error(matrix[mask], reconstructed[mask])) if np.sum(mask) > 0 else 0
            
            # Calculate coverage (non-zero predictions)
            coverage = np.sum(reconstructed > 0.1) / reconstructed.size
            
            metrics = {
                'rmse': rmse,
                'coverage': coverage,
                'reconstruction_error': model.reconstruction_err_,
                'n_components': model.n_components
            }
            
            # Add target achievement metrics
            metrics['coverage_target_achievement'] = coverage / 0.942  # Target user coverage
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating NMF metrics: {e}")
            return {'rmse': float('inf'), 'coverage': 0}
    
    async def train_all_models(self) -> Dict[str, Any]:
        """Train all models and return results"""
        logger.info("Starting comprehensive model training...")
        
        # Load training data
        interactions_df, user_item_matrix = self.load_training_data()
        
        results = {}
        
        # Train SVD model
        try:
            svd_model, svd_metrics = self.train_svd_model(user_item_matrix)
            results['svd'] = {
                'model': svd_model,
                'metrics': svd_metrics,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"SVD training failed: {e}")
            results['svd'] = {'status': 'failed', 'error': str(e)}
        
        # Train NMF model
        try:
            nmf_model, nmf_metrics = self.train_nmf_model(user_item_matrix)
            results['nmf'] = {
                'model': nmf_model,
                'metrics': nmf_metrics,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"NMF training failed: {e}")
            results['nmf'] = {'status': 'failed', 'error': str(e)}
        
        # Generate comprehensive evaluation report
        evaluation_report = self._generate_evaluation_report(results)
        results['evaluation_report'] = evaluation_report
        
        logger.info("Model training completed")
        return results
    
    def _generate_evaluation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            'training_timestamp': time.time(),
            'target_metrics': self.target_metrics,
            'model_performance': {},
            'recommendations': []
        }
        
        for model_name, model_data in results.items():
            if model_name == 'evaluation_report':
                continue
                
            if model_data['status'] == 'success':
                metrics = model_data['metrics']
                performance = {}
                
                for metric_name, target_value in self.target_metrics.items():
                    if metric_name in metrics:
                        current_value = metrics[metric_name]
                        achievement_rate = (current_value / target_value) if target_value > 0 else 0
                        meets_target = current_value >= target_value
                        
                        performance[metric_name] = {
                            'current': current_value,
                            'target': target_value,
                            'achievement_rate': achievement_rate,
                            'meets_target': meets_target
                        }
                
                report['model_performance'][model_name] = performance
                
                # Generate recommendations
                if model_name == 'svd':
                    if metrics.get('rmse', float('inf')) <= self.target_metrics['rmse']:
                        report['recommendations'].append(f"✓ SVD model meets RMSE target ({metrics['rmse']:.4f} ≤ {self.target_metrics['rmse']})")
                    else:
                        report['recommendations'].append(f"✗ SVD model needs improvement for RMSE ({metrics['rmse']:.4f} > {self.target_metrics['rmse']})")
                
                elif model_name == 'nmf':
                    if metrics.get('coverage', 0) >= 0.9:
                        report['recommendations'].append(f"✓ NMF model provides good coverage ({metrics['coverage']:.3f})")
                    else:
                        report['recommendations'].append(f"✗ NMF model needs better coverage ({metrics['coverage']:.3f})")
        
        return report

# CLI interface for model training
async def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train recommendation models')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--models', nargs='+', choices=['svd', 'nmf', 'all'], default=['all'], help='Models to train')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    try:
        # Train models
        results = await trainer.train_all_models()
        
        # Print results
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        
        for model_name, model_data in results.items():
            if model_name == 'evaluation_report':
                continue
                
            print(f"\n{model_name.upper()} Model:")
            if model_data['status'] == 'success':
                metrics = model_data['metrics']
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
            else:
                print(f"  Status: {model_data['status']}")
                if 'error' in model_data:
                    print(f"  Error: {model_data['error']}")
        
        # Print evaluation report
        if 'evaluation_report' in results:
            report = results['evaluation_report']
            print(f"\nEVALUATION REPORT")
            print("-" * 40)
            for recommendation in report.get('recommendations', []):
                print(f"  {recommendation}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
