"""
Kafka Producer for Real-time Data Streaming
Handles user interactions and model updates with high throughput
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
from kafka import KafkaProducer as SyncKafkaProducer
from kafka.errors import KafkaError
import structlog

logger = structlog.get_logger()

class KafkaProducer:
    """Async Kafka producer for real-time event streaming"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config['bootstrap_servers']
        self.producer = None
        self._initialize_producer()
    
    def _initialize_producer(self):
        """Initialize Kafka producer with optimal configuration"""
        try:
            self.producer = SyncKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: str(x).encode('utf-8') if x else None,
                # Performance optimizations
                batch_size=16384,  # 16KB batches
                linger_ms=10,      # Wait up to 10ms for batching
                compression_type='snappy',
                acks='1',          # Wait for leader acknowledgment
                retries=3,
                max_in_flight_requests_per_connection=5,
                # Timeout configurations
                request_timeout_ms=30000,
                metadata_max_age_ms=300000,
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def send_message(
        self, 
        topic: str, 
        message: Dict[str, Any], 
        key: Optional[str] = None
    ) -> bool:
        """Send message to Kafka topic asynchronously"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
            
            # Send message
            future = self.producer.send(topic, value=message, key=key)
            
            # Wait for delivery in background
            await asyncio.get_event_loop().run_in_executor(
                None, future.get, 10  # 10 second timeout
            )
            
            logger.debug(f"Message sent to topic {topic}", key=key)
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending message to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to {topic}: {e}")
            return False
    
    async def send_user_interaction(self, interaction: Dict[str, Any]) -> bool:
        """Send user interaction event"""
        return await self.send_message(
            'user_interactions',
            interaction,
            key=str(interaction.get('user_id'))
        )
    
    async def send_recommendation_served(self, recommendation_data: Dict[str, Any]) -> bool:
        """Send recommendation served event"""
        return await self.send_message(
            'recommendations_served',
            recommendation_data,
            key=str(recommendation_data.get('user_id'))
        )
    
    async def send_model_update(self, model_data: Dict[str, Any]) -> bool:
        """Send model update event"""
        return await self.send_message(
            'model_updates',
            model_data,
            key=model_data.get('model_id')
        )
    
    def close(self):
        """Close the producer and flush pending messages"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")

class KafkaConsumer:
    """Kafka consumer for processing real-time events"""
    
    def __init__(self, config: Dict[str, Any], topics: list):
        self.config = config
        self.topics = topics
        self.consumer = None
        self.running = False
    
    def _initialize_consumer(self):
        """Initialize Kafka consumer"""
        from kafka import KafkaConsumer as SyncKafkaConsumer
        
        try:
            self.consumer = SyncKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config['bootstrap_servers'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                group_id='recommendation_engine',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=500,
                fetch_min_bytes=1024,
                fetch_max_wait_ms=500
            )
            logger.info(f"Kafka consumer initialized for topics: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    async def start_consuming(self, message_handler):
        """Start consuming messages"""
        self._initialize_consumer()
        self.running = True
        
        logger.info("Starting Kafka consumer...")
        
        try:
            while self.running:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if message_batch:
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            try:
                                await message_handler(
                                    topic=message.topic,
                                    key=message.key,
                                    value=message.value,
                                    offset=message.offset,
                                    timestamp=message.timestamp
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing message from {message.topic}: {e}",
                                    key=message.key,
                                    offset=message.offset
                                )
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")

# Message schemas for different event types
class MessageSchemas:
    """Standard message schemas for different event types"""
    
    USER_INTERACTION = {
        "user_id": int,
        "item_id": int,
        "rating": float,
        "interaction_type": str,  # "click", "view", "purchase", "rating"
        "timestamp": float,
        "session_id": str,
        "context": dict  # Additional context (device, location, etc.)
    }
    
    RECOMMENDATION_SERVED = {
        "user_id": int,
        "recommendations": list,  # List of item_ids
        "algorithm": str,
        "timestamp": float,
        "request_id": str,
        "response_time_ms": float
    }
    
    MODEL_UPDATE = {
        "model_id": str,
        "model_type": str,  # "svd", "nmf", "hybrid"
        "version": str,
        "metrics": dict,
        "timestamp": float,
        "status": str  # "training", "deployed", "retired"
    }

# Example usage
async def example_producer_usage():
    """Example of how to use the Kafka producer"""
    
    config = {
        'bootstrap_servers': ['localhost:9092']
    }
    
    producer = KafkaProducer(config)
    
    # Send user interaction
    interaction = {
        "user_id": 12345,
        "item_id": 67890,
        "rating": 4.5,
        "interaction_type": "rating",
        "session_id": "session_abc123"
    }
    
    success = await producer.send_user_interaction(interaction)
    print(f"Interaction sent: {success}")
    
    # Send recommendation served
    recommendation = {
        "user_id": 12345,
        "recommendations": [1, 2, 3, 4, 5],
        "algorithm": "hybrid",
        "request_id": "req_xyz789",
        "response_time_ms": 45.2
    }
    
    success = await producer.send_recommendation_served(recommendation)
    print(f"Recommendation event sent: {success}")
    
    producer.close()

async def example_consumer_usage():
    """Example of how to use the Kafka consumer"""
    
    config = {
        'bootstrap_servers': ['localhost:9092']
    }
    
    async def handle_message(topic, key, value, offset, timestamp):
        """Handle incoming messages"""
        print(f"Received message from {topic}: {value}")
        
        if topic == 'user_interactions':
            # Process user interaction
            print(f"Processing interaction for user {value['user_id']}")
        elif topic == 'recommendations_served':
            # Log recommendation serving
            print(f"Recommendation served to user {value['user_id']}")
    
    consumer = KafkaConsumer(
        config, 
        ['user_interactions', 'recommendations_served']
    )
    
    try:
        await consumer.start_consuming(handle_message)
    except KeyboardInterrupt:
        consumer.stop()

if __name__ == "__main__":
    # Run producer example
    asyncio.run(example_producer_usage())
    
    # To run consumer, uncomment the line below
    # asyncio.run(example_consumer_usage())
