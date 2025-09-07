#!/usr/bin/env python3
"""
Setup script for Real-Time Recommendation Engine
Initializes databases, creates tables, and sets up the environment
"""

import os
import sys
import time
import subprocess
import asyncio
from pathlib import Path
import yaml
import psycopg2
import redis
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

def print_step(step: str):
    """Print setup step"""
    print(f"\n🔧 {step}...")

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message: str):
    """Print warning message"""
    print(f"⚠️ {message}")

class SetupManager:
    """Manages the setup process for the recommendation engine"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "config" / "config.yaml"
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print_error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print_step("Checking dependencies")
        
        required_packages = [
            'pyspark', 'delta-spark', 'mlflow', 'kafka-python',
            'fastapi', 'uvicorn', 'redis', 'psycopg2-binary'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ✗ {package}")
        
        if missing_packages:
            print_error(f"Missing packages: {', '.join(missing_packages)}")
            print("Install them with: pip install -r requirements.txt")
            return False
        
        print_success("All dependencies are installed")
        return True
    
    def check_docker_services(self):
        """Check if Docker services are running"""
        print_step("Checking Docker services")
        
        services = {
            'PostgreSQL': ('localhost', 5432),
            'Redis': ('localhost', 6379),
            'Kafka': ('localhost', 9092)
        }
        
        all_running = True
        
        for service_name, (host, port) in services.items():
            if self.check_port(host, port):
                print(f"  ✓ {service_name} ({host}:{port})")
            else:
                print(f"  ✗ {service_name} ({host}:{port})")
                all_running = False
        
        if not all_running:
            print_warning("Some services are not running. Start them with:")
            print("  docker-compose up -d")
            return False
        
        print_success("All Docker services are running")
        return True
    
    def check_port(self, host: str, port: int) -> bool:
        """Check if a port is open"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                return result == 0
        except:
            return False
    
    def setup_postgresql(self):
        """Setup PostgreSQL database and tables"""
        print_step("Setting up PostgreSQL database")
        
        db_config = self.config['database']['postgres']
        
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['username'],
                password=db_config['password']
            )
            
            cursor = conn.cursor()
            
            # Create tables
            self.create_postgres_tables(cursor)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print_success("PostgreSQL database setup completed")
            return True
            
        except Exception as e:
            print_error(f"PostgreSQL setup failed: {e}")
            return False
    
    def create_postgres_tables(self, cursor):
        """Create necessary PostgreSQL tables"""
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                user_features JSONB
            )
        """)
        
        # Items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                item_id SERIAL PRIMARY KEY,
                category VARCHAR(100),
                price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                item_features JSONB
            )
        """)
        
        # User interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                interaction_id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id),
                item_id INTEGER REFERENCES items(item_id),
                interaction_type VARCHAR(50),
                rating DECIMAL(3,2),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(100),
                context JSONB
            )
        """)
        
        # Experiments table for A/B testing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR(100) PRIMARY KEY,
                name VARCHAR(200),
                config JSONB,
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                metric_id SERIAL PRIMARY KEY,
                model_name VARCHAR(100),
                model_version VARCHAR(100),
                metric_name VARCHAR(100),
                metric_value DECIMAL(10,6),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("  ✓ Created database tables")
    
    def setup_redis(self):
        """Setup Redis cache"""
        print_step("Setting up Redis cache")
        
        redis_config = self.config['database']['redis']
        
        try:
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                password=redis_config.get('password')
            )
            
            # Test connection
            r.ping()
            
            # Clear any existing data (optional)
            # r.flushdb()
            
            print_success("Redis cache setup completed")
            return True
            
        except Exception as e:
            print_error(f"Redis setup failed: {e}")
            return False
    
    def setup_delta_lake(self):
        """Setup Delta Lake tables"""
        print_step("Setting up Delta Lake tables")
        
        try:
            # Initialize Spark with Delta Lake
            builder = SparkSession.builder.appName("Setup") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            spark = configure_spark_with_delta_pip(builder).getOrCreate()
            
            # Create Delta Lake directory
            delta_path = "/tmp/delta-tables"
            os.makedirs(delta_path, exist_ok=True)
            
            # Create sample data and tables
            self.create_delta_tables(spark, delta_path)
            
            spark.stop()
            print_success("Delta Lake tables setup completed")
            return True
            
        except Exception as e:
            print_error(f"Delta Lake setup failed: {e}")
            return False
    
    def create_delta_tables(self, spark, delta_path: str):
        """Create Delta Lake tables with sample data"""
        import pandas as pd
        from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType, TimestampType
        
        # User interactions table
        interactions_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("item_id", LongType(), True),
            StructField("rating", DoubleType(), True),
            StructField("interaction_type", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("session_id", StringType(), True)
        ])
        
        # Create sample interactions data
        sample_data = []
        for i in range(10000):
            sample_data.append((
                int(i % 1000),  # user_id
                int(i % 500),   # item_id
                float(3.0 + (i % 3)),  # rating
                "rating",       # interaction_type
                float(time.time()),  # timestamp
                f"session_{i % 100}"  # session_id
            ))
        
        interactions_df = spark.createDataFrame(sample_data, interactions_schema)
        
        # Write to Delta Lake
        interactions_df.write.format("delta") \
            .mode("overwrite") \
            .save(f"{delta_path}/interactions")
        
        print("  ✓ Created interactions table")
        
        # User profiles table
        user_profiles_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("avg_rating", DoubleType(), True),
            StructField("interaction_count", LongType(), True),
            StructField("last_interaction", DoubleType(), True)
        ])
        
        # Create sample user profiles
        user_data = []
        for i in range(1000):
            user_data.append((
                int(i),
                float(3.5 + (i % 2) * 0.5),
                int(10 + i % 50),
                float(time.time())
            ))
        
        user_profiles_df = spark.createDataFrame(user_data, user_profiles_schema)
        user_profiles_df.write.format("delta") \
            .mode("overwrite") \
            .save(f"{delta_path}/user_profiles")
        
        print("  ✓ Created user profiles table")
    
    def setup_kafka_topics(self):
        """Setup Kafka topics"""
        print_step("Setting up Kafka topics")
        
        topics = [
            'user_interactions',
            'recommendations_served',
            'model_updates',
            'ab_test_events'
        ]
        
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            from kafka.errors import TopicAlreadyExistsError
            
            admin_client = KafkaAdminClient(
                bootstrap_servers=['localhost:9092'],
                client_id='setup_client'
            )
            
            topic_list = []
            for topic in topics:
                topic_list.append(NewTopic(
                    name=topic,
                    num_partitions=3,
                    replication_factor=1
                ))
            
            try:
                admin_client.create_topics(new_topics=topic_list, validate_only=False)
                print("  ✓ Created Kafka topics")
            except TopicAlreadyExistsError:
                print("  ✓ Kafka topics already exist")
            
            print_success("Kafka topics setup completed")
            return True
            
        except Exception as e:
            print_error(f"Kafka topics setup failed: {e}")
            return False
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        print_step("Setting up MLflow tracking")
        
        try:
            import mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            
            # Create experiment
            try:
                experiment_id = mlflow.create_experiment(self.config['mlflow']['experiment_name'])
                print(f"  ✓ Created MLflow experiment: {experiment_id}")
            except:
                print("  ✓ MLflow experiment already exists")
            
            # Create artifacts directory
            artifacts_dir = Path(self.config['mlflow']['artifact_location'])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Created artifacts directory")
            
            print_success("MLflow tracking setup completed")
            return True
            
        except Exception as e:
            print_error(f"MLflow setup failed: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print_step("Creating project directories")
        
        directories = [
            "data",
            "logs",
            "models",
            "artifacts",
            "checkpoints",
            "/tmp/delta-tables",
            "/tmp/spark-checkpoint",
            "/tmp/mlflow-artifacts"
        ]
        
        for directory in directories:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {directory}")
        
        print_success("Project directories created")
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("🚀 Setting up Real-Time Recommendation Engine\n")
        
        steps = [
            ("Check dependencies", self.check_dependencies),
            ("Check Docker services", self.check_docker_services),
            ("Create directories", self.create_directories),
            ("Setup PostgreSQL", self.setup_postgresql),
            ("Setup Redis", self.setup_redis),
            ("Setup Kafka topics", self.setup_kafka_topics),
            ("Setup Delta Lake", self.setup_delta_lake),
            ("Setup MLflow", self.setup_mlflow)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print_error(f"{step_name} failed: {e}")
                failed_steps.append(step_name)
        
        # Summary
        print("\n" + "="*60)
        if failed_steps:
            print_warning(f"Setup completed with {len(failed_steps)} issues:")
            for step in failed_steps:
                print(f"  ❌ {step}")
            print("\nPlease resolve the issues above before running the system.")
        else:
            print_success("🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Start the recommendation API:")
            print("   python src/api/recommendation_api.py")
            print("\n2. Start feature processing (optional):")
            print("   python src/streaming/feature_processor.py")
            print("\n3. Run the demo:")
            print("   python run_demo.py")
        
        print("="*60)

def main():
    """Main setup function"""
    setup_manager = SetupManager()
    setup_manager.run_full_setup()

if __name__ == "__main__":
    main()
