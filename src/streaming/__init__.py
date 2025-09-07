"""
Real-time streaming components
Kafka-based event processing and feature engineering
"""

from .kafka_producer import KafkaProducer
from .feature_processor import FeatureProcessor

__all__ = ["KafkaProducer", "FeatureProcessor"]
