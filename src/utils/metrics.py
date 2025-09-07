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
                if len(true_lists) > 0
