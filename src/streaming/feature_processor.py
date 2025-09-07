"""
Real-time Feature Processing with PySpark Streaming
Processes user interactions and updates features with 67% dimensionality reduction
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from delta.tables import DeltaTable
import structlog
import yaml

logger = structlog.get_logger()

class FeatureProcessor:
    """Real-time feature processing with advanced dimensionality reduction"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.spark = self._initialize_spark()
        self.feature_pipeline = None
        self.feature_stats = {}
        
        # Feature engineering parameters
        self.target_variance = self.config['features']['dimensionality_reduction']['target_variance']
        self.max_components = self.config['features']['dimensionality_reduction']['max_components']
        
    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session with optimal configuration"""
        spark = SparkSession.builder \
            .appName("FeatureProcessor") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.streaming.kafka.consumer.cache.enabled", "false") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/feature-checkpoint") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
    
    def create_feature_pipeline(self) -> Pipeline:
        """Create ML pipeline for feature engineering with dimensionality reduction"""
        
        # Define feature columns (this would be based on your actual schema)
        feature_columns = [
            "user_age", "user_gender", "user_location", "item_category",
            "item_price", "rating_avg", "rating_count", "interaction_count",
            "time_since_last_interaction", "session_duration", "click_count",
            "view_count", "purchase_count", "cart_adds", "page_views"
        ]
        
        # Vector assembler
        vector_assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="raw_features"
        )
        
        # Standard scaler
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # PCA for dimensionality reduction (67% reduction target)
        original_dims = len(feature_columns)
        target_dims = int(original_dims * (1 - 0.67))  # 67% reduction
        
        pca = PCA(
            k=min(target_dims, self.max_components),
            inputCol="scaled_features",
            outputCol="pca_features"
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[vector_assembler, scaler, pca])
        
        logger.info(f"Created feature pipeline: {original_dims} -> {target_dims} dimensions")
        return pipeline
    
    def process_user_interactions_stream(self):
        """Process real-time user interactions from Kafka"""
        
        # Define schema for user interactions
        interaction_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("item_id", LongType(), True),
            StructField("rating", DoubleType(), True),
            StructField("interaction_type", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("session_id", StringType(), True),
            StructField("context", MapType(StringType(), StringType()), True)
        ])
        
        # Read from Kafka
        kafka_config = self.config['streaming']['kafka']
        interactions_stream = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_config['bootstrap_servers']) \
            .option("subscribe", "user_interactions") \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON messages
        parsed_interactions = interactions_stream.select(
            from_json(col("value").cast("string"), interaction_schema).alias("data")
        ).select("data.*")
        
        # Add derived features
        enriched_interactions = self._add_derived_features(parsed_interactions)
        
        # Process in micro-batches
        query = enriched_interactions.writeStream \
            .foreachBatch(self._process_interaction_batch) \
            .outputMode("append") \
            .trigger(processingTime='10 seconds') \
            .start()
        
        return query
    
    def _add_derived_features(self, df: DataFrame) -> DataFrame:
        """Add derived features to interaction data"""
        
        # Convert timestamp to proper datetime
        df = df.withColumn("interaction_time", from_unixtime(col("timestamp")))
        
        # Time-based features
        df = df.withColumn("hour_of_day", hour(col("interaction_time"))) \
               .withColumn("day_of_week", dayofweek(col("interaction_time"))) \
               .withColumn("is_weekend", when(dayofweek(col("interaction_time")).isin([1, 7]), 1).otherwise(0))
        
        # Session-based features (would require windowing in real implementation)
        df = df.withColumn("session_duration", lit(0.0))  # Placeholder
        df = df.withColumn("interactions_in_session", lit(1))  # Placeholder
        
        # User behavior features (would require lookups to user profile tables)
        df = df.withColumn("user_avg_rating", lit(3.5))  # Placeholder
        df = df.withColumn("user_interaction_count", lit(100))  # Placeholder
        
        # Item features (would require lookups to item catalog)
        df = df.withColumn("item_popularity", lit(0.5))  # Placeholder
        df = df.withColumn("item_avg_rating", lit(4.0))  # Placeholder
        
        return df
    
    def _process_interaction_batch(self, df: DataFrame, epoch_id: int):
        """Process each micro-batch of interactions"""
        try:
            logger.info(f"Processing batch {epoch_id} with {df.count()} records")
            
            if df.count() == 0:
                return
            
            # Enrich with user and item features
            enriched_df = self._enrich_with_features(df)
            
            # Apply feature engineering pipeline
            if self.feature_pipeline is None:
                self.feature_pipeline = self._train_feature_pipeline(enriched_df)
            
            processed_df = self.feature_pipeline.transform(enriched_df)
            
            # Update user profiles in Delta Lake
            self._update_user_profiles(processed_df)
            
            # Update item features
            self._update_item_features(processed_df)
            
            # Calculate feature statistics
            self._update_feature_stats(processed_df)
            
        except Exception as e:
            logger.error(f"Error processing batch {epoch_id}: {e}")
    
    def _enrich_with_features(self, df: DataFrame) -> DataFrame:
        """Enrich interactions with user and item features"""
        
        # Mock user features (in production, would join with user profile table)
        user_features = df.select("user_id").distinct() \
            .withColumn("user_age", (rand() * 50 + 18).cast("int")) \
            .withColumn("user_gender", when(rand() > 0.5, "M").otherwise("F")) \
            .withColumn("user_location", lit("US")) \
            .withColumn("user_tenure_days", (rand() * 365).cast("int"))
        
        # Mock item features (in production, would join with item catalog)
        item_features = df.select("item_id").distinct() \
            .withColumn("item_category", lit("electronics")) \
            .withColumn("item_price", rand() * 1000) \
            .withColumn("item_brand", lit("Brand_A")) \
            .withColumn("item_age_days", (rand() * 365).cast("int"))
        
        # Join features
        enriched_df = df \
            .join(user_features, "user_id", "left") \
            .join(item_features, "item_id", "left")
        
        return enriched_df
    
    def _train_feature_pipeline(self, df: DataFrame) -> Pipeline:
        """Train the feature engineering pipeline"""
        logger.info("Training feature engineering pipeline")
        
        # Select numeric features for the pipeline
        feature_columns = [
            "user_age", "user_tenure_days", "item_price", "item_age_days",
            "rating", "hour_of_day", "day_of_week", "is_weekend"
        ]
        
        # Create and train pipeline
        pipeline = self._create_ml_pipeline(feature_columns)
        trained_pipeline = pipeline.fit(df)
        
        # Calculate variance explained by PCA
        pca_model = trained_pipeline.stages[-1]  # Last stage is PCA
        explained_variance = pca_model.explainedVariance.toArray()
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info(f"PCA variance explained: {cumulative_variance[-1]:.3f}")
        logger.info(f"Dimensionality reduction: {len(feature_columns)} -> {len(explained_variance)}")
        
        return trained_pipeline
    
    def _create_ml_pipeline(self, feature_columns: List[str]) -> Pipeline:
        """Create ML pipeline for feature processing"""
        
        # Vector assembler
        vector_assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="raw_features"
        )
        
        # Standard scaler
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # PCA for dimensionality reduction
        target_components = max(1, int(len(feature_columns) * 0.33))  # 67% reduction
        
        pca = PCA(
            k=target_components,
            inputCol="scaled_features",
            outputCol="pca_features"
        )
        
        return Pipeline(stages=[vector_assembler, scaler, pca])
    
    def _update_user_profiles(self, df: DataFrame):
        """Update user profiles in Delta Lake"""
        try:
            # Aggregate user features
            user_profiles = df.groupBy("user_id") \
                .agg(
                    avg("rating").alias("avg_rating"),
                    count("*").alias("interaction_count"),
                    max("timestamp").alias("last_interaction"),
                    collect_set("item_id").alias("interacted_items")
                )
            
            # Write to Delta Lake (merge/upsert)
            user_profiles.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .save("/delta/user_profiles")
            
            logger.debug(f"Updated {user_profiles.count()} user profiles")
            
        except Exception as e:
            logger.error(f"Error updating user profiles: {e}")
    
    def _update_item_features(self, df: DataFrame):
        """Update item features in Delta Lake"""
        try:
            # Aggregate item features
            item_features = df.groupBy("item_id") \
                .agg(
                    avg("rating").alias("avg_rating"),
                    count("*").alias("interaction_count"),
                    countDistinct("user_id").alias("unique_users"),
                    max("timestamp").alias("last_interaction")
                )
            
            # Write to Delta Lake
            item_features.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .save("/delta/item_features")
            
            logger.debug(f"Updated {item_features.count()} item features")
            
        except Exception as e:
            logger.error(f"Error updating item features: {e}")
    
    def _update_feature_stats(self, df: DataFrame):
        """Update feature statistics for monitoring"""
        try:
            stats = df.describe().collect()
            
            # Store statistics for monitoring
            for row in stats:
                metric_name = row['summary']
                for col_name in df.columns:
                    if col_name in row.asDict():
                        value = row[col_name]
                        if value is not None:
                            self.feature_stats[f"{col_name}_{metric_name}"] = float(value)
            
            logger.debug("Updated feature statistics")
            
        except Exception as e:
            logger.error(f"Error updating feature stats: {e}")
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get current feature statistics"""
        return {
            'feature_stats': self.feature_stats,
            'pipeline_trained': self.feature_pipeline is not None,
            'target_variance': self.target_variance,
            'dimensionality_reduction': 0.67
        }
    
    async def start_streaming(self):
        """Start the feature processing stream"""
        logger.info("Starting feature processing stream...")
        
        try:
            # Start processing user interactions
            query = self.process_user_interactions_stream()
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        finally:
            self.spark.stop()

# Example usage
async def main():
    """Main function to start feature processing"""
    processor = FeatureProcessor()
    
    try:
        await processor.start_streaming()
    except KeyboardInterrupt:
        logger.info("Stopping feature processor...")
    except Exception as e:
        logger.error(f"Feature processor error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
