# Real-Time Recommendation Engine Makefile
# Convenient commands for development and deployment

.PHONY: help install setup start stop test clean deploy demo docs

# Default target
help:
	@echo "Real-Time Recommendation Engine"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  setup       - Setup environment and initialize services"
	@echo "  start       - Start all services"
	@echo "  stop        - Stop all services"
	@echo "  test        - Run tests"
	@echo "  demo        - Run demonstration"
	@echo "  train       - Train recommendation models"
	@echo "  clean       - Clean up temporary files"
	@echo "  deploy      - Deploy to production"
	@echo "  docs        - Generate documentation"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"

# Installation and setup
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

setup: install
	@echo "Setting up environment..."
	python scripts/setup.py
	@echo "✅ Environment setup complete"

# Service management
start-infra:
	@echo "Starting infrastructure services..."
	docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	sleep 10
	@echo "✅ Infrastructure services started"

start-api:
	@echo "Starting recommendation API..."
	python src/api/recommendation_api.py &
	@echo "✅ API started on http://localhost:8000"

start-streaming:
	@echo "Starting feature processing..."
	python src/streaming/feature_processor.py &
	@echo "✅ Streaming processor started"

start: start-infra start-api
	@echo "🚀 All services started successfully!"
	@echo "API available at: http://localhost:8000"
	@echo "MLflow UI: http://localhost:5000"
	@echo "Grafana: http://localhost:3000"

stop:
	@echo "Stopping all services..."
	docker-compose down
	pkill -f "recommendation_api.py" || true
	pkill -f "feature_processor.py" || true
	@echo "✅ All services stopped"

# Model training
train:
	@echo "Training recommendation models..."
	python src/models/train_models.py
	@echo "✅ Model training completed"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests completed"

test-api:
	@echo "Testing API endpoints..."
	pytest tests/api/ -v
	@echo "✅ API tests completed"

test-models:
	@echo "Testing models..."
	pytest tests/models/ -v
	@echo "✅ Model tests completed"

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v
	@echo "✅ Integration tests completed"

# Demo and benchmarks
demo:
	@echo "Running demonstration..."
	python run_demo.py
	@echo "✅ Demo completed"

benchmark:
	@echo "Running performance benchmarks..."
	python scripts/benchmark.py
	@echo "✅ Benchmarks completed"

ab-test:
	@echo "Running A/B test demonstration..."
	python src/experiments/ab_testing.py
	@echo "✅ A/B test completed"

# Code quality
lint:
	@echo "Running code linting..."
	flake8 src/ tests/ --max-line-length=100
	pylint src/ --rcfile=.pylintrc
	@echo "✅ Linting completed"

format:
	@echo "Formatting code..."
	black src/ tests/ --line-length=100
	isort src/ tests/
	@echo "✅ Code formatted"

type-check:
	@echo "Running type checking..."
	mypy src/ --ignore-missing-imports
	@echo "✅ Type checking completed"

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/
	@echo "✅ Documentation generated"

docs-serve:
	@echo "Serving documentation..."
	cd docs/_build && python -m http.server 8080
	@echo "📚 Documentation available at http://localhost:8080"

# Data and cleanup
generate-data:
	@echo "Generating sample data..."
	python scripts/generate_sample_data.py
	@echo "✅ Sample data generated"

clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf /tmp/spark-checkpoint/
	rm -rf /tmp/delta-warehouse/
	@echo "✅ Cleanup completed"

# Monitoring and health checks
health:
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "❌ API not responding"
	curl -f http://localhost:5000/ || echo "❌ MLflow not responding"
	redis-cli ping || echo "❌ Redis not responding"
	@echo "✅ Health check completed"

logs:
	@echo "Showing service logs..."
	docker-compose logs -f

metrics:
	@echo "Showing system metrics..."
	curl -s http://localhost:8000/metrics
	@echo "✅ Metrics retrieved"

# Development helpers
dev-setup: install
	@echo "Setting up development environment..."
	pre-commit install
	pip install -e .
	@echo "✅ Development environment ready"

jupyter:
	@echo "Starting Jupyter notebook..."
	jupyter notebook notebooks/
	@echo "📓 Jupyter started"

shell:
	@echo "Starting interactive shell..."
	python -i scripts/interactive_shell.py

# Deployment
deploy-staging:
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.staging.yml up -d
	@echo "✅ Staging deployment completed"

deploy-prod:
	@echo "Deploying to production..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Production deployment completed"

# Database operations
db-migrate:
	@echo "Running database migrations..."
	python scripts/migrate_db.py
	@echo "✅ Database migration completed"

db-seed:
	@echo "Seeding database with sample data..."
	python scripts/seed_database.py
	@echo "✅ Database seeded"

db-backup:
	@echo "Creating database backup..."
	docker exec recommendation-engine_postgres_1 pg_dump -U postgres recommendations > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backup created"

# Model operations
model-register:
	@echo "Registering models in MLflow..."
	python scripts/register_models.py
	@echo "✅ Models registered"

model-deploy:
	@echo "Deploying models to production..."
	python scripts/deploy_models.py
	@echo "✅ Models deployed"

model-validate:
	@echo "Validating model performance..."
	python scripts/validate_models.py
	@echo "✅ Model validation completed"

# Quick commands for common workflows
quick-start: install start-infra
	@sleep 5
	@make start-api
	@echo "🚀 Quick start completed! API ready at http://localhost:8000"

full-setup: setup start train
	@echo "🎉 Full setup completed with trained models!"

ci: lint type-check test
	@echo "✅ CI pipeline completed successfully"

# Help for specific components
help-api:
	@echo "API Commands:"
	@echo "  start-api   - Start the recommendation API"
	@echo "  test-api    - Test API endpoints"
	@echo "  health      - Check API health"

help-models:
	@echo "Model Commands:"
	@echo "  train          - Train all models"
	@echo "  model-register - Register models in MLflow"
	@echo "  model-deploy   - Deploy models"
	@echo "  model-validate - Validate model performance"

help-data:
	@echo "Data Commands:"
	@echo "  generate-data - Generate sample data"
	@echo "  db-seed      - Seed database"
	@echo "  db-migrate   - Run migrations"
	@echo "  db-backup    - Backup database"
