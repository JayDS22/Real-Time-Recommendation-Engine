"""
PyTest configuration and fixtures for testing
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
import asyncio
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def sample_user_item_matrix():
    """Generate sample user-item interaction matrix"""
    np.random.seed(42)
    n_users, n_items = 100, 50
    density = 0.1
    
    matrix = np.zeros((n_users, n_items))
    n_interactions = int(n_users * n_items * density)
    
    for _ in range(n_interactions):
        user_idx = np.random.randint(0, n_users)
        item_idx = np.random.randint(0, n_items)
        rating = np.random.normal(3.5, 1.0)
        matrix[user_idx, item_idx] = np.clip(rating, 1, 5)
    
    return matrix

@pytest.fixture
def sample_interactions_df():
    """Generate sample interactions DataFrame"""
    np.random.seed(42)
    n_interactions = 1000
    
    data = []
    for i in range(n_interactions):
        data.append({
            'user_id': np.random.randint(1, 101),
            'item_id': np.random.randint(1, 51),
            'rating': np.clip(np.random.normal(3.5, 1.0), 1, 5),
            'timestamp': 1640995200.0 + i * 3600,  # Start from 2022-01-01
            'interaction_type': np.random.choice(['rating', 'view', 'purchase']),
            'session_id': f"session_{i % 50}"
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_recommendations():
    """Generate sample recommendations for testing"""
    return {
        123: [
            {'item_id': 1, 'score': 0.95, 'algorithm': 'svd'},
            {'item_id': 2, 'score': 0.87, 'algorithm': 'svd'},
            {'item_id': 3, 'score': 0.82, 'algorithm': 'svd'}
        ],
        456: [
            {'item_id': 4, 'score': 0.91, 'algorithm': 'nmf'},
            {'item_id': 5, 'score': 0.88, 'algorithm': 'nmf'},
            {'item_id': 6, 'score': 0.84, 'algorithm': 'nmf'}
        ]
    }

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'models': {
            'svd': {
                'factors': 10,
                'learning_rate': 0.01,
                'regularization': 0.1,
                'epochs': 10
            },
            'nmf': {
                'factors': 5,
                'alpha': 0.0001,
                'l1_ratio': 0.0,
                'max_iter': 20
            }
        },
        'api': {
            'host': 'localhost',
            'port': 8000,
            'cache_ttl': 300
        },
        'database': {
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 1  # Use different DB for testing
            }
        },
        'evaluation': {
            'metrics': ['ndcg', 'map', 'hit_rate', 'rmse'],
            'k_values': [5, 10],
            'test_size': 0.2
        }
    }

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async testing"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_kafka_config():
    """Mock Kafka configuration for testing"""
    return {
        'bootstrap_servers': ['localhost:9092'],
        'topics': {
            'user_interactions': 'test_user_interactions',
            'recommendations': 'test_recommendations'
        }
    }

@pytest.fixture
def sample_ab_test_data():
    """Sample A/B test data"""
    np.random.seed(42)
    
    # Control group (lower CTR)
    control_data = np.random.binomial(1, 0.025, 1000)  # 2.5% CTR
    
    # Treatment group (higher CTR - 23% lift)
    treatment_data = np.random.binomial(1, 0.025 * 1.23, 1000)  # ~3.1% CTR
    
    return control_data, treatment_data

@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing"""
    return {
        'ndcg_10': 0.78,
        'map_10': 0.73,
        'hit_rate_20': 0.91,
        'rmse': 0.84,
        'user_coverage': 0.942,
        'catalog_coverage': 0.785,
        'r2_score': 0.89
    }
