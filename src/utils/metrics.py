"""
Recommendation System Metrics
Comprehensive evaluation metrics including NDCG, MAP, Hit Rate, Coverage, and RMSE
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import rankdata
import structlog
from collections import defaultdict
import time

logger = structlog.get_logger()

class RecommendationMetrics:
    """Comprehensive metrics calculator for recommendation systems"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.calculation_times = {}
    
    def calculate_ndcg(
        self, 
        y_true: List[float], 
        y_pred: List[float], 
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k)
        Target: 0.78 for NDCG@10
        """
        start_time = time.time()
        
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0
            
            # Ensure we have the same length
            min_len = min(len(y_true), len(y_pred), k)
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate DCG
            dcg = self._calculate_dcg(y_true, y_pred, k)
            
            # Calculate IDCG (Ideal DCG)
            ideal_order = sorted(y_true, reverse=True)
            idcg = self._calculate_dcg(ideal_order, ideal_order, k)
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            self.calculation_times['ndcg'] = time.time() - start_time
            return ndcg
            
        except Exception as e:
            logger.error(f"Error calculating NDCG: {e}")
            return 0.0
    
    def _calculate_dcg(self, y_true: List[float], y_pred: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain"""
        dcg = 0.0
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        
        for i, idx in enumerate(sorted_indices[:k]):
            relevance = y_true[idx]
            dcg += (2**relevance - 1) / np.log2(i + 2)
        
        return dcg
    
    def calculate_map(
        self, 
        y_true_list: List[List[int]], 
        y_pred_list: List[List[int]], 
        k: int = 10
    ) -> float:
        """
        Calculate Mean Average Precision (MAP@k)
        Target: 0.73 for MAP@10
        """
        start_time = time.time()
        
        try:
            if len(y_true_list) == 0 or len(y_pred_list) == 0:
                return 0.0
            
            total_ap = 0.0
            valid_queries = 0
            
            for y_true, y_pred in zip(y_true_list, y_pred_list):
                ap = self._calculate_average_precision(y_true, y_pred, k)
                if ap is not None:
                    total_ap += ap
                    valid_queries += 1
            
            map_score = total_ap / valid_queries if valid_queries > 0 else 0.0
            
            self.calculation_times['map'] = time.time() - start_time
            return map_score
            
        except Exception as e:
            logger.error(f"Error calculating MAP: {e}")
            return 0.0
    
    def _calculate_average_precision(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        k: int
    ) -> Optional[float]:
        """Calculate Average Precision for a single query"""
        if len(y_true) == 0:
            return None
        
        # Take top k predictions
        y_pred_k = y_pred[:k]
        
        # Calculate precision at each relevant position
        num_relevant = 0
        sum_precision = 0.0
        
        for i, item in enumerate(y_pred_k):
            if item in y_true:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i
        
        # Average precision
        total_relevant = len(y_true)
        if total_relevant == 0:
            return 0.0
        
        return sum_precision / min(total_relevant, k)
    
    def calculate_hit_rate(
        self, 
        y_true_list: List[List[int]], 
        y_pred_list: List[List[int]], 
        k: int = 20
    ) -> float:
        """
        Calculate Hit Rate (Recall@k)
        Target: 0.91 for Hit Rate@20
        """
        start_time = time.time()
        
        try:
            if len(y_true_list) == 0 or len(y_pred_list) == 0:
                return 0.0
            
            hits = 0
            total_queries = len(y_true_list)
            
            for y_true, y_pred in zip(y_true_list, y_pred_list):
                y_pred_k = set(y_pred[:k])
                y_true_set = set(y_true)
                
                # Check if there's any intersection
                if len(y_pred_k.intersection(y_true_set)) > 0:
                    hits += 1
            
            hit_rate = hits / total_queries if total_queries > 0 else 0.0
            
            self.calculation_times['hit_rate'] = time.time() - start_time
            return hit_rate
            
        except Exception as e:
            logger.error(f"Error calculating Hit Rate: {e}")
            return 0.0
    
    def calculate_coverage(
        self, 
        recommendations: List[List[int]], 
        total_items: int
    ) -> float:
        """
        Calculate Catalog Coverage
        Target: 94.2% user coverage, 78.5% catalog coverage
        """
        start_time = time.time()
        
        try:
            if not recommendations or total_items <= 0:
                return 0.0
            
            # Get all unique recommended items
            recommended_items = set()
            for rec_list in recommendations:
                recommended_items.update(rec_list)
            
            coverage = len(recommended_items) / total_items
            
            self.calculation_times['coverage'] = time.time() - start_time
            return coverage
            
        except Exception as e:
            logger.error(f"Error calculating coverage: {e}")
            return 0.0
    
    def calculate_user_coverage(
        self, 
        user_recommendations: Dict[int, List[int]], 
        total_users: int
    ) -> float:
        """Calculate User Coverage (percentage of users who received recommendations)"""
        try:
            users_with_recs = len(user_recommendations)
            return users_with_recs / total_users if total_users > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating user coverage: {e}")
            return 0.0
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error
        Target: 0.84 RMSE
        """
        start_time = time.time()
        
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return float('inf')
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            self.calculation_times['rmse'] = time.time() - start_time
            return rmse
            
        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}")
            return float('inf')
    
    def calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² Score for prediction accuracy
        Target: 0.89 R² score
        """
        start_time = time.time()
        
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0
            
            r2 = r2_score(y_true, y_pred)
            
            self.calculation_times['r2_score'] = time.time() - start_time
            return r2
            
        except Exception as e:
            logger.error(f"Error calculating R² score: {e}")
            return 0.0
    
    def calculate_precision_at_k(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        k: int = 10
    ) -> float:
        """Calculate Precision@k"""
        try:
            if k <= 0 or len(y_pred) == 0:
                return 0.0
            
            y_pred_k = y_pred[:k]
            y_true_set = set(y_true)
            
            relevant_retrieved = len([item for item in y_pred_k if item in y_true_set])
            
            return relevant_retrieved / k
            
        except Exception as e:
            logger.error(f"Error calculating Precision@{k}: {e}")
            return 0.0
    
    def calculate_recall_at_k(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        k: int = 10
    ) -> float:
        """Calculate Recall@k"""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0
            
            y_pred_k = set(y_pred[:k])
            y_true_set = set(y_true)
            
            relevant_retrieved = len(y_pred_k.intersection(y_true_set))
            
            return relevant_retrieved / len(y_true_set)
            
        except Exception as e:
            logger.error(f"Error calculating Recall@{k}: {e}")
            return 0.0
    
    def calculate_diversity(self, recommendations: List[List[int]]) -> float:
        """Calculate diversity of recommendations using Intra-List Diversity"""
        try:
            if not recommendations:
                return 0.0
            
            total_diversity = 0.0
            valid_lists = 0
            
            for rec_list in recommendations:
                if len(rec_list) > 1:
                    # Calculate pairwise diversity (simplified)
                    unique_items = len(set(rec_list))
                    diversity = unique_items / len(rec_list)
                    total_diversity += diversity
                    valid_lists += 1
            
            return total_diversity / valid_lists if valid_lists > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0
    
    def calculate_novelty(
        self, 
        recommendations: List[List[int]], 
        item_popularity: Dict[int, float]
    ) -> float:
        """Calculate novelty using item popularity"""
        try:
            if not recommendations or not item_popularity:
                return 0.0
            
            total_novelty = 0.0
            total_items = 0
            
            for rec_list in recommendations:
                for item in rec_list:
                    if item in item_popularity:
                        # Novelty is inverse of popularity
                        novelty = 1.0 - item_popularity[item]
                        total_novelty += novelty
                        total_items += 1
            
            return total_novelty / total_items if total_items > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 0.0
    
    def evaluate_recommendations(
        self,
        true_ratings: np.ndarray,
        predicted_ratings: np.ndarray,
        user_item_recommendations: Dict[int, List[int]],
        user_item_ground_truth: Dict[int, List[int]],
        total_items: int,
        total_users: int,
        item_popularity: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of recommendation system
        Returns all key metrics matching the project specifications
        """
        logger.info("Starting comprehensive recommendation evaluation...")
        
        metrics = {}
        
        # Rating prediction metrics
        if len(true_ratings) > 0 and len(predicted_ratings) > 0:
            metrics['rmse'] = self.calculate_rmse(true_ratings, predicted_ratings)
            metrics['r2_score'] = self.calculate_r2_score(true_ratings, predicted_ratings)
        
        # Ranking metrics
        if user_item_recommendations and user_item_ground_truth:
            # Prepare data for ranking metrics
            true_lists = []
            pred_lists = []
            
            for user_id in user_item_recommendations:
                if user_id in user_item_ground_truth:
                    true_lists.append(user_item_ground_truth[user_id])
                    pred_lists.append(user_item_recommendations[user_id])
            
            if true_lists and pred_lists:
                # Calculate ranking metrics
                metrics['hit_rate_20'] = self.calculate_hit_rate(true_lists, pred_lists, k=20)
                metrics['map_10'] = self.calculate_map(true_lists, pred_lists, k=10)
                
                # Calculate NDCG (requires ratings, approximated here)
                if len(true_lists) > 0:
                    # Approximate NDCG using binary relevance
                    ndcg_scores = []
                    for true_items, pred_items in zip(true_lists, pred_lists):
                        # Create binary relevance scores
                        true_relevance = [1.0 if item in true_items else 0.0 for item in pred_items[:10]]
                        pred_relevance = [1.0] * len(pred_items[:10])  # All predicted items have score 1
                        
                        if len(true_relevance) > 0:
                            ndcg = self.calculate_ndcg(true_relevance, pred_relevance, k=10)
                            ndcg_scores.append(ndcg)
                    
                    metrics['ndcg_10'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
                
                # Precision and Recall metrics
                precision_scores = []
                recall_scores = []
                
                for true_items, pred_items in zip(true_lists, pred_lists):
                    precision_scores.append(self.calculate_precision_at_k(true_items, pred_items, k=10))
                    recall_scores.append(self.calculate_recall_at_k(true_items, pred_items, k=10))
                
                metrics['precision_10'] = np.mean(precision_scores) if precision_scores else 0.0
                metrics['recall_10'] = np.mean(recall_scores) if recall_scores else 0.0
        
        # Coverage metrics
        if user_item_recommendations:
            all_recommendations = list(user_item_recommendations.values())
            metrics['catalog_coverage'] = self.calculate_coverage(all_recommendations, total_items)
            metrics['user_coverage'] = self.calculate_user_coverage(user_item_recommendations, total_users)
        
        # Diversity and Novelty
        if user_item_recommendations:
            all_recommendations = list(user_item_recommendations.values())
            metrics['diversity'] = self.calculate_diversity(all_recommendations)
            
            if item_popularity:
                metrics['novelty'] = self.calculate_novelty(all_recommendations, item_popularity)
        
        # Log computation times
        for metric_name, computation_time in self.calculation_times.items():
            logger.info(f"{metric_name} computation time: {computation_time:.4f}s")
        
        logger.info("Recommendation evaluation completed", metrics=metrics)
        return metrics

class MetricsCollector:
    """Collects and aggregates metrics for monitoring"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.current_metrics = {}
        self.target_metrics = {
            'ndcg_10': 0.78,
            'map_10': 0.73,
            'hit_rate_20': 0.91,
            'rmse': 0.84,
            'user_coverage': 0.942,
            'catalog_coverage': 0.785,
            'r2_score': 0.89
        }
    
    def record_metrics(self, metrics: Dict[str, float], timestamp: Optional[float] = None):
        """Record metrics with timestamp"""
        if timestamp is None:
            timestamp = time.time()
        
        self.current_metrics = metrics.copy()
        
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return self.current_metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of metrics performance"""
        summary = {}
        
        for metric_name, target_value in self.target_metrics.items():
            current_value = self.current_metrics.get(metric_name, 0.0)
            
            summary[metric_name] = {
                'current': current_value,
                'target': target_value,
                'achievement_rate': (current_value / target_value) if target_value > 0 else 0.0,
                'meets_target': current_value >= target_value
            }
        
        return summary
    
    def get_performance_trends(self, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """Get performance trends for a specific metric"""
        if metric_name not in self.metrics_history:
            return {}
        
        history = self.metrics_history[metric_name]
        recent_history = history[-window_size:] if len(history) >= window_size else history
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data'}
        
        values = [entry['value'] for entry in recent_history]
        
        # Calculate trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        trend_direction = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'current_value': values[-1],
            'average_value': np.mean(values),
            'std_deviation': np.std(values),
            'min_value': min(values),
            'max_value': max(values)
        }

# Utility functions for specific metric calculations
def calculate_ndcg(y_true: List[float], y_pred: List[float], k: int = 10) -> float:
    """Standalone NDCG calculation function"""
    calculator = RecommendationMetrics()
    return calculator.calculate_ndcg(y_true, y_pred, k)

def calculate_map(y_true_list: List[List[int]], y_pred_list: List[List[int]], k: int = 10) -> float:
    """Standalone MAP calculation function"""
    calculator = RecommendationMetrics()
    return calculator.calculate_map(y_true_list, y_pred_list, k)

def calculate_hit_rate(y_true_list: List[List[int]], y_pred_list: List[List[int]], k: int = 20) -> float:
    """Standalone Hit Rate calculation function"""
    calculator = RecommendationMetrics()
    return calculator.calculate_hit_rate(y_true_list, y_pred_list, k)

def calculate_coverage(recommendations: List[List[int]], total_items: int) -> float:
    """Standalone Coverage calculation function"""
    calculator = RecommendationMetrics()
    return calculator.calculate_coverage(recommendations, total_items)

# Example usage and testing
def generate_sample_data() -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Generate sample data for testing metrics"""
    np.random.seed(42)
    
    # Sample rating data
    n_ratings = 10000
    true_ratings = np.random.normal(3.5, 1.0, n_ratings)
    true_ratings = np.clip(true_ratings, 1, 5)
    
    # Predictions with some noise
    predicted_ratings = true_ratings + np.random.normal(0, 0.5, n_ratings)
    predicted_ratings = np.clip(predicted_ratings, 1, 5)
    
    # Sample recommendation data
    n_users = 1000
    n_items = 5000
    
    user_recommendations = {}
    user_ground_truth = {}
    
    for user_id in range(n_users):
        # Generate recommendations (top 20 items)
        recommended_items = np.random.choice(n_items, size=20, replace=False).tolist()
        user_recommendations[user_id] = recommended_items
        
        # Generate ground truth (items user actually liked)
        # Some overlap with recommendations for realistic metrics
        ground_truth_size = np.random.randint(5, 15)
        overlap_size = np.random.randint(0, min(5, ground_truth_size))
        
        ground_truth = []
        if overlap_size > 0:
            ground_truth.extend(np.random.choice(recommended_items[:10], size=overlap_size, replace=False))
        
        remaining_size = ground_truth_size - overlap_size
        if remaining_size > 0:
            available_items = [i for i in range(n_items) if i not in ground_truth]
            ground_truth.extend(np.random.choice(available_items, size=remaining_size, replace=False))
        
        user_ground_truth[user_id] = ground_truth
    
    return true_ratings, predicted_ratings, user_recommendations, user_ground_truth

def test_metrics():
    """Test all metrics with sample data"""
    print("Testing Recommendation Metrics...")
    
    # Generate sample data
    true_ratings, pred_ratings, user_recs, user_truth = generate_sample_data()
    
    # Initialize metrics calculator
    metrics_calc = RecommendationMetrics()
    
    # Calculate comprehensive metrics
    results = metrics_calc.evaluate_recommendations(
        true_ratings=true_ratings,
        predicted_ratings=pred_ratings,
        user_item_recommendations=user_recs,
        user_item_ground_truth=user_truth,
        total_items=5000,
        total_users=1000
    )
    
    print("\n" + "="*50)
    print("RECOMMENDATION SYSTEM METRICS")
    print("="*50)
    
    target_metrics = {
        'NDCG@10': (results.get('ndcg_10', 0), 0.78),
        'MAP@10': (results.get('map_10', 0), 0.73),
        'Hit Rate@20': (results.get('hit_rate_20', 0), 0.91),
        'RMSE': (results.get('rmse', 0), 0.84),
        'User Coverage': (results.get('user_coverage', 0), 0.942),
        'Catalog Coverage': (results.get('catalog_coverage', 0), 0.785),
        'R² Score': (results.get('r2_score', 0), 0.89)
    }
    
    for metric_name, (actual, target) in target_metrics.items():
        achievement = (actual / target * 100) if target > 0 else 0
        status = "✓" if actual >= target else "✗"
        print(f"{metric_name:15} | {actual:.4f} | Target: {target:.4f} | {achievement:6.1f}% {status}")
    
    print("="*50)
    
    return results

if __name__ == "__main__":
    # Run tests
    test_results = test_metrics()
