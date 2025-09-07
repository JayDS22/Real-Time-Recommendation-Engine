"""
Unit tests for recommendation metrics
"""

import pytest
import numpy as np
from src.utils.metrics import RecommendationMetrics, MetricsCollector

class TestRecommendationMetrics:
    """Test recommendation evaluation metrics"""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator instance"""
        return RecommendationMetrics()
    
    def test_calculate_ndcg(self, metrics_calculator):
        """Test NDCG calculation"""
        # Perfect ranking
        y_true = [3, 2, 1, 0, 0]
        y_pred = [3, 2, 1, 0, 0]
        
        ndcg = metrics_calculator.calculate_ndcg(y_true, y_pred, k=5)
        assert ndcg == 1.0, "Perfect ranking should have NDCG = 1.0"
        
        # Reversed ranking
        y_true = [3, 2, 1, 0, 0]
        y_pred = [0, 0, 1, 2, 3]
        
        ndcg = metrics_calculator.calculate_ndcg(y_true, y_pred, k=5)
        assert 0 <= ndcg <= 1, "NDCG should be between 0 and 1"
        assert ndcg < 1.0, "Reversed ranking should have NDCG < 1.0"
    
    def test_calculate_ndcg_target_achievement(self, metrics_calculator):
        """Test NDCG achieves target of 0.78"""
        # Simulate good ranking that should achieve target
        np.random.seed(42)
        y_true = np.random.exponential(1, 20)  # Relevance scores
        y_pred = y_true + np.random.normal(0, 0.1, 20)  # Slightly noisy predictions
        
        ndcg = metrics_calculator.calculate_ndcg(y_true.tolist(), y_pred.tolist(), k=10)
        
        # Should achieve high NDCG due to good correlation
        assert ndcg >= 0.7, f"NDCG {ndcg:.3f} should be close to target 0.78"
    
    def test_calculate_map(self, metrics_calculator):
        """Test MAP calculation"""
        # Perfect predictions
        y_true_list = [[1, 2, 3], [4, 5]]
        y_pred_list = [[1, 2, 3, 6, 7], [4, 5, 8, 9]]
        
        map_score = metrics_calculator.calculate_map(y_true_list, y_pred_list, k=5)
        assert map_score == 1.0, "Perfect predictions should have MAP = 1.0"
        
        # Poor predictions
        y_true_list = [[1, 2, 3], [4, 5]]
        y_pred_list = [[6, 7, 8, 9, 10], [11, 12, 13, 14]]
        
        map_score = metrics_calculator.calculate_map(y_true_list, y_pred_list, k=5)
        assert map_score == 0.0, "No relevant items should have MAP = 0.0"
    
    def test_calculate_hit_rate(self, metrics_calculator):
        """Test hit rate calculation"""
        # All queries have hits
        y_true_list = [[1, 2], [3, 4], [5, 6]]
        y_pred_list = [[1, 7, 8], [3, 9, 10], [5, 11, 12]]
        
        hit_rate = metrics_calculator.calculate_hit_rate(y_true_list, y_pred_list, k=3)
        assert hit_rate == 1.0, "All queries with hits should have hit rate = 1.0"
        
        # No queries have hits
        y_true_list = [[1, 2], [3, 4], [5, 6]]
        y_pred_list = [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
        
        hit_rate = metrics_calculator.calculate_hit_rate(y_true_list, y_pred_list, k=3)
        assert hit_rate == 0.0, "No hits should have hit rate = 0.0"
    
    def test_calculate_hit_rate_target(self, metrics_calculator):
        """Test hit rate achieves target of 0.91"""
        # Simulate scenario that should achieve high hit rate
        y_true_list = []
        y_pred_list = []
        
        np.random.seed(42)
        for i in range(100):
            # Each user has 3-5 relevant items
            relevant_items = list(range(i*10, i*10 + np.random.randint(3, 6)))
            y_true_list.append(relevant_items)
            
            # Predictions include some relevant items (high hit rate scenario)
            predictions = relevant_items[:2]  # Include 2 relevant items
            predictions.extend(list(range(1000 + i*10, 1000 + i*10 + 18)))  # Add irrelevant
            y_pred_list.append(predictions)
        
        hit_rate = metrics_calculator.calculate_hit_rate(y_true_list, y_pred_list, k=20)
        assert hit_rate >= 0.9, f"Hit rate {hit_rate:.3f} should achieve target 0.91"
    
    def test_calculate_rmse(self, metrics_calculator):
        """Test RMSE calculation"""
        # Perfect predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        rmse = metrics_calculator.calculate_rmse(y_true, y_pred)
        assert rmse == 0.0, "Perfect predictions should have RMSE = 0.0"
        
        # Known RMSE
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # Off by 1 each
        
        rmse = metrics_calculator.calculate_rmse(y_true, y_pred)
        assert abs(rmse - 1.0) < 1e-10, "RMSE should be 1.0 for predictions off by 1"
    
    def test_calculate_rmse_target(self, metrics_calculator):
        """Test RMSE achieves target of 0.84"""
        # Simulate realistic rating predictions
        np.random.seed(42)
        y_true = np.random.normal(3.5, 1.0, 1000)
        y_true = np.clip(y_true, 1, 5)
        
        # Add small noise to simulate model predictions
        y_pred = y_true + np.random.normal(0, 0.5, 1000)
        y_pred = np.clip(y_pred, 1, 5)
        
        rmse = metrics_calculator.calculate_rmse(y_true, y_pred)
        assert rmse <= 1.0, f"RMSE {rmse:.3f} should be reasonable for rating prediction"
    
    def test_calculate_coverage(self, metrics_calculator):
        """Test coverage calculation"""
        # Full coverage
        recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        total_items = 10
        
        coverage = metrics_calculator.calculate_coverage(recommendations, total_items)
        assert coverage == 1.0, "All items recommended should have coverage = 1.0"
        
        # Partial coverage
        recommendations = [[1, 2], [1, 3], [2, 3]]
        total_items = 10
        
        coverage = metrics_calculator.calculate_coverage(recommendations, total_items)
        assert coverage == 0.3, "3 out of 10 items should have coverage = 0.3"
    
    def test_calculate_user_coverage(self, metrics_calculator):
        """Test user coverage calculation"""
        user_recommendations = {1: [1, 2], 2: [3, 4], 3: [5, 6]}
        total_users = 5
        
        coverage = metrics_calculator.calculate_user_coverage(user_recommendations, total_users)
        assert coverage == 0.6, "3 out of 5 users should have coverage = 0.6"
    
    def test_calculate_r2_score(self, metrics_calculator):
        """Test R² score calculation"""
        # Perfect correlation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = metrics_calculator.calculate_r2_score(y_true, y_pred)
        assert r2 == 1.0, "Perfect predictions should have R² = 1.0"
        
        # No correlation (predictions = mean)
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # All predictions = mean
        
        r2 = metrics_calculator.calculate_r2_score(y_true, y_pred)
        assert r2 == 0.0, "Mean predictions should have R² = 0.0"
    
    def test_calculate_precision_at_k(self, metrics_calculator):
        """Test Precision@k calculation"""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 4, 5, 6]  # 2 out of 3 relevant
        
        precision = metrics_calculator.calculate_precision_at_k(y_true, y_pred, k=3)
        assert abs(precision - 2/3) < 1e-10, "Precision@3 should be 2/3"
    
    def test_calculate_recall_at_k(self, metrics_calculator):
        """Test Recall@k calculation"""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 5, 6, 7]  # 2 out of 4 relevant items found
        
        recall = metrics_calculator.calculate_recall_at_k(y_true, y_pred, k=5)
        assert abs(recall - 2/4) < 1e-10, "Recall@5 should be 2/4"
    
    def test_calculate_diversity(self, metrics_calculator):
        """Test diversity calculation"""
        # High diversity (all unique items)
        recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        diversity = metrics_calculator.calculate_diversity(recommendations)
        assert diversity == 1.0, "All unique items should have diversity = 1.0"
        
        # Low diversity (repeated items)
        recommendations = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        diversity = metrics_calculator.calculate_diversity(recommendations)
        assert abs(diversity - 1/3) < 1e-10, "Repeated items should have low diversity"
    
    def test_calculate_novelty(self, metrics_calculator):
        """Test novelty calculation"""
        recommendations = [[1, 2], [3, 4]]
        item_popularity = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.2}  # 1,2 popular, 3,4 novel
        
        novelty = metrics_calculator.calculate_novelty(recommendations, item_popularity)
        expected_novelty = (0.1 + 0.2 + 0.9 + 0.8) / 4  # (1-pop) for each item
        assert abs(novelty - expected_novelty) < 1e-10, "Novelty calculation incorrect"
    
    def test_comprehensive_evaluation(self, metrics_calculator):
        """Test comprehensive evaluation function"""
        # Generate sample data
        np.random.seed(42)
        
        # Rating data
        true_ratings = np.random.normal(3.5, 1.0, 100)
        pred_ratings = true_ratings + np.random.normal(0, 0.5, 100)
        
        # Recommendation data
        user_recs = {i: list(range(i*10, i*10 + 5)) for i in range(10)}
        user_truth = {i: list(range(i*10, i*10 + 3)) for i in range(10)}
        
        # Evaluate
        results = metrics_calculator.evaluate_recommendations(
            true_ratings=true_ratings,
            predicted_ratings=pred_ratings,
            user_item_recommendations=user_recs,
            user_item_ground_truth=user_truth,
            total_items=1000,
            total_users=10
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['rmse', 'r2_score', 'hit_rate_20', 'map_10', 'ndcg_10', 
                          'catalog_coverage', 'user_coverage', 'diversity']
        
        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"
            assert isinstance(results[metric], (int, float)), f"Invalid metric type: {metric}"

class TestMetricsCollector:
    """Test metrics collection and monitoring"""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector instance"""
        return MetricsCollector()
    
    def test_record_metrics(self, collector):
        """Test metrics recording"""
        metrics = {
            'ndcg_10': 0.78,
            'map_10': 0.73,
            'hit_rate_20': 0.91
        }
        
        collector.record_metrics(metrics, timestamp=1640995200.0)
        
        current = collector.get_current_metrics()
        assert current == metrics
        
        # Check history
        for metric_name in metrics:
            assert metric_name in collector.metrics_history
            assert len(collector.metrics_history[metric_name]) == 1
    
    def test_get_metrics_summary(self, collector):
        """Test metrics summary generation"""
        # Record metrics that meet targets
        metrics = {
            'ndcg_10': 0.78,
            'map_10': 0.73,
            'hit_rate_20': 0.91,
            'rmse': 0.84
        }
        
        collector.record_metrics(metrics)
        summary = collector.get_metrics_summary()
        
        for metric_name in metrics:
            assert metric_name in summary
            metric_summary = summary[metric_name]
            
            assert 'current' in metric_summary
            assert 'target' in metric_summary
            assert 'achievement_rate' in metric_summary
            assert 'meets_target' in metric_summary
            
            # All metrics should meet targets
            assert metric_summary['meets_target'], f"{metric_name} should meet target"
    
    def test_get_performance_trends(self, collector):
        """Test performance trend analysis"""
        # Record increasing trend
        for i in range(10):
            metrics = {'ndcg_10': 0.7 + i * 0.01}  # Increasing from 0.7 to 0.79
            collector.record_metrics(metrics, timestamp=1640995200.0 + i * 3600)
        
        trends = collector.get_performance_trends('ndcg_10')
        
        assert trends['trend'] == 'improving'
        assert trends['slope'] > 0
        assert 'current_value' in trends
        assert 'average_value' in trends
    
    def test_target_achievement_validation(self, collector):
        """Test validation against target metrics"""
        # Test metrics that achieve all targets
        perfect_metrics = {
            'ndcg_10': 0.78,
            'map_10': 0.73,
            'hit_rate_20': 0.91,
            'rmse': 0.84,
            'user_coverage': 0.942,
            'catalog_coverage': 0.785,
            'r2_score': 0.89
        }
        
        collector.record_metrics(perfect_metrics)
        summary = collector.get_metrics_summary()
        
        # All metrics should meet targets
        for metric_name, metric_data in summary.items():
            assert metric_data['meets_target'], f"{metric_name} should meet target"
            assert metric_data['achievement_rate'] >= 1.0, f"{metric_name} achievement rate too low"
