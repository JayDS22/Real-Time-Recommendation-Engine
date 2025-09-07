"""
A/B Testing Framework for Recommendation Systems
Implements statistical testing with power analysis and effect size measurement
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
import mlflow
import structlog
from dataclasses import dataclass, asdict
import yaml

logger = structlog.get_logger()

@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiments"""
    name: str
    description: str
    control_algorithm: str
    treatment_algorithm: str
    metric: str
    min_effect_size: float
    significance_level: float
    statistical_power: float
    traffic_split: float
    max_duration_days: int
    min_sample_size: int

@dataclass
class ExperimentResult:
    """Results of an A/B test experiment"""
    experiment_id: str
    start_time: datetime
    end_time: datetime
    control_size: int
    treatment_size: int
    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    is_significant: bool
    lift_percentage: float
    recommendation: str

class StatisticalAnalyzer:
    """Statistical analysis utilities for A/B testing"""
    
    @staticmethod
    def calculate_sample_size(
        effect_size: float,
        power: float = 0.95,
        alpha: float = 0.05,
        two_sided: bool = True
    ) -> int:
        """Calculate required sample size for given effect size and power"""
        from scipy.stats import norm
        
        if two_sided:
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)
        
        z_beta = norm.ppf(power)
        
        # Sample size calculation for two-sample test
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    @staticmethod
    def calculate_effect_size(control_mean: float, treatment_mean: float, pooled_std: float) -> float:
        """Calculate Cohen's d effect size"""
        if pooled_std == 0:
            return 0
        return (treatment_mean - control_mean) / pooled_std
    
    @staticmethod
    def calculate_confidence_interval(
        diff: float, 
        std_error: float, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the difference"""
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        margin_error = z_score * std_error
        
        return (diff - margin_error, diff + margin_error)
    
    @staticmethod
    def perform_statistical_test(
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        test_type: str = "ttest"
    ) -> Tuple[float, float]:
        """Perform statistical test and return test statistic and p-value"""
        if test_type == "ttest":
            statistic, p_value = ttest_ind(treatment_data, control_data)
        elif test_type == "mann_whitney":
            from scipy.stats import mannwhitneyu
            statistic, p_value = mannwhitneyu(treatment_data, control_data, alternative='two-sided')
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return statistic, p_value

class ABTestFramework:
    """Comprehensive A/B testing framework for recommendation systems"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ab_config = self.config['ab_testing']
        self.analyzer = StatisticalAnalyzer()
        self.active_experiments = {}
        self.experiment_results = {}
        
        # MLflow setup for experiment tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment("ab_testing")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        control_algorithm: str,
        treatment_algorithm: str,
        metric: str = "ctr",
        min_effect_size: Optional[float] = None,
        significance_level: Optional[float] = None,
        statistical_power: Optional[float] = None,
        traffic_split: Optional[float] = None,
        max_duration_days: Optional[int] = None
    ) -> str:
        """Create a new A/B test experiment"""
        
        experiment_config = ExperimentConfig(
            name=name,
            description=description,
            control_algorithm=control_algorithm,
            treatment_algorithm=treatment_algorithm,
            metric=metric,
            min_effect_size=min_effect_size or self.ab_config['min_effect_size'],
            significance_level=significance_level or self.ab_config['significance_level'],
            statistical_power=statistical_power or self.ab_config['statistical_power'],
            traffic_split=traffic_split or self.ab_config['default_traffic_split'],
            max_duration_days=max_duration_days or self.ab_config['max_experiment_duration'],
            min_sample_size=self.ab_config['min_sample_size']
        )
        
        # Calculate required sample size
        required_sample_size = self.analyzer.calculate_sample_size(
            effect_size=experiment_config.min_effect_size,
            power=experiment_config.statistical_power,
            alpha=experiment_config.significance_level
        )
        
        experiment_id = str(uuid.uuid4())
        
        # Store experiment configuration
        self.active_experiments[experiment_id] = {
            'config': experiment_config,
            'required_sample_size': required_sample_size,
            'start_time': datetime.now(),
            'control_data': [],
            'treatment_data': [],
            'user_assignments': {}
        }
        
        logger.info(
            "Created A/B test experiment",
            experiment_id=experiment_id,
            name=name,
            required_sample_size=required_sample_size
        )
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"ab_test_{name}"):
            mlflow.log_params(asdict(experiment_config))
            mlflow.log_metric("required_sample_size", required_sample_size)
        
        return experiment_id
    
    def assign_user_to_variant(self, experiment_id: str, user_id: int) -> str:
        """Assign user to control or treatment group"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        
        # Check if user already assigned
        if user_id in experiment['user_assignments']:
            return experiment['user_assignments'][user_id]
        
        # Hash-based assignment for consistency
        np.random.seed(hash(f"{experiment_id}_{user_id}") % (2**32))
        variant = "treatment" if np.random.random() < experiment['config'].traffic_split else "control"
        
        experiment['user_assignments'][user_id] = variant
        
        return variant
    
    def record_metric(self, experiment_id: str, user_id: int, metric_value: float):
        """Record metric value for a user in the experiment"""
        if experiment_id not in self.active_experiments:
            return
        
        experiment = self.active_experiments[experiment_id]
        variant = self.assign_user_to_variant(experiment_id, user_id)
        
        if variant == "control":
            experiment['control_data'].append(metric_value)
        else:
            experiment['treatment_data'].append(metric_value)
        
        logger.debug(
            "Recorded metric",
            experiment_id=experiment_id,
            user_id=user_id,
            variant=variant,
            metric_value=metric_value
        )
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results and determine statistical significance"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        config = experiment['config']
        
        control_data = np.array(experiment['control_data'])
        treatment_data = np.array(experiment['treatment_data'])
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            raise ValueError("Insufficient data for analysis")
        
        # Calculate means
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        
        # Calculate pooled standard deviation
        pooled_var = ((len(control_data) - 1) * np.var(control_data, ddof=1) + 
                     (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) / \
                    (len(control_data) + len(treatment_data) - 2)
        pooled_std = np.sqrt(pooled_var)
        
        # Effect size (Cohen's d)
        effect_size = self.analyzer.calculate_effect_size(control_mean, treatment_mean, pooled_std)
        
        # Statistical test
        test_statistic, p_value = self.analyzer.perform_statistical_test(control_data, treatment_data)
        
        # Confidence interval
        std_error = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        diff = treatment_mean - control_mean
        confidence_interval = self.analyzer.calculate_confidence_interval(diff, std_error)
        
        # Calculate actual statistical power (post-hoc)
        from scipy.stats import norm
        z_score = abs(test_statistic)
        actual_power = 1 - norm.cdf(norm.ppf(1 - config.significance_level/2) - z_score)
        
        # Determine significance
        is_significant = p_value < config.significance_level
        
        # Calculate lift percentage
        if control_mean != 0:
            lift_percentage = ((treatment_mean - control_mean) / control_mean) * 100
        else:
            lift_percentage = 0
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_significant, effect_size, config.min_effect_size, lift_percentage
        )
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            start_time=experiment['start_time'],
            end_time=datetime.now(),
            control_size=len(control_data),
            treatment_size=len(treatment_data),
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            statistical_power=actual_power,
            is_significant=is_significant,
            lift_percentage=lift_percentage,
            recommendation=recommendation
        )
        
        # Store result
        self.experiment_results[experiment_id] = result
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"ab_result_{config.name}"):
            mlflow.log_metrics({
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "effect_size": effect_size,
                "p_value": p_value,
                "statistical_power": actual_power,
                "lift_percentage": lift_percentage,
                "control_size": len(control_data),
                "treatment_size": len(treatment_data)
            })
            mlflow.log_param("is_significant", is_significant)
            mlflow.log_param("recommendation", recommendation)
        
        logger.info(
            "Experiment analysis completed",
            experiment_id=experiment_id,
            p_value=p_value,
            effect_size=effect_size,
            lift_percentage=lift_percentage,
            is_significant=is_significant
        )
        
        return result
    
    def _generate_recommendation(
        self, 
        is_significant: bool, 
        effect_size: float, 
        min_effect_size: float,
        lift_percentage: float
    ) -> str:
        """Generate actionable recommendation based on results"""
        if not is_significant:
            return "No significant difference detected. Consider longer experiment duration or larger sample size."
        
        if effect_size < min_effect_size:
            return "Statistically significant but effect size below practical threshold. Consider if implementation cost is justified."
        
        if lift_percentage > 0:
            return f"Treatment shows significant improvement ({lift_percentage:.1f}% lift). Recommend implementing treatment algorithm."
        else:
            return f"Treatment shows significant decrease ({abs(lift_percentage):.1f}% drop). Recommend keeping control algorithm."
    
    def run_experiment(self, experiment_id: str, duration_days: Optional[int] = None) -> ExperimentResult:
        """Run complete A/B test experiment with simulated data"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        config = experiment['config']
        
        # Use provided duration or default from config
        duration = duration_days or config.max_duration_days
        
        logger.info(f"Running experiment {experiment_id} for {duration} days")
        
        # Simulate experiment data (replace with real data collection in production)
        self._simulate_experiment_data(experiment_id, duration)
        
        # Analyze results
        result = self.analyze_experiment(experiment_id)
        
        return result
    
    def _simulate_experiment_data(self, experiment_id: str, duration_days: int):
        """Simulate A/B test data for demonstration purposes"""
        experiment = self.active_experiments[experiment_id]
        config = experiment['config']
        
        # Simulate daily user interactions
        users_per_day = max(100, experiment['required_sample_size'] // duration_days)
        
        # Control group metrics (baseline CTR around 2.5%)
        control_base_rate = 0.025
        
        # Treatment group with the target 23% lift
        treatment_multiplier = 1.23  # 23% lift
        treatment_base_rate = control_base_rate * treatment_multiplier
        
        total_control_users = 0
        total_treatment_users = 0
        
        for day in range(duration_days):
            daily_users = int(users_per_day * (1 + 0.1 * np.random.randn()))  # Add some variance
            
            for _ in range(daily_users):
                user_id = np.random.randint(1, 1000000)
                variant = self.assign_user_to_variant(experiment_id, user_id)
                
                if variant == "control":
                    # Simulate control group metric (binary outcome for CTR)
                    metric_value = 1 if np.random.random() < control_base_rate else 0
                    total_control_users += 1
                else:
                    # Simulate treatment group metric with lift
                    metric_value = 1 if np.random.random() < treatment_base_rate else 0
                    total_treatment_users += 1
                
                self.record_metric(experiment_id, user_id, metric_value)
        
        logger.info(
            f"Simulated {total_control_users} control users and {total_treatment_users} treatment users"
        )
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        config = experiment['config']
        
        control_size = len(experiment['control_data'])
        treatment_size = len(experiment['treatment_data'])
        total_size = control_size + treatment_size
        
        progress = min(total_size / experiment['required_sample_size'], 1.0)
        
        # Calculate interim metrics if we have data
        interim_analysis = None
        if control_size > 10 and treatment_size > 10:
            try:
                interim_analysis = self.analyze_experiment(experiment_id)
            except:
                pass
        
        status = {
            'experiment_id': experiment_id,
            'name': config.name,
            'status': 'running' if progress < 1.0 else 'ready_for_analysis',
            'progress': progress,
            'control_size': control_size,
            'treatment_size': treatment_size,
            'required_sample_size': experiment['required_sample_size'],
            'days_running': (datetime.now() - experiment['start_time']).days,
            'max_duration_days': config.max_duration_days,
            'interim_analysis': interim_analysis
        }
        
        return status
    
    def stop_experiment(self, experiment_id: str) -> ExperimentResult:
        """Stop an experiment and return final results"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        logger.info(f"Stopping experiment {experiment_id}")
        
        # Perform final analysis
        result = self.analyze_experiment(experiment_id)
        
        # Move from active to completed
        del self.active_experiments[experiment_id]
        
        return result
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments (active and completed)"""
        experiments = {}
        
        # Active experiments
        for exp_id, exp_data in self.active_experiments.items():
            experiments[exp_id] = {
                'status': 'active',
                'config': exp_data['config'],
                'start_time': exp_data['start_time'],
                'current_status': self.get_experiment_status(exp_id)
            }
        
        # Completed experiments
        for exp_id, result in self.experiment_results.items():
            experiments[exp_id] = {
                'status': 'completed',
                'result': result
            }
        
        return experiments

# Example usage and demo
async def demo_ab_testing():
    """Demonstration of the A/B testing framework"""
    
    # Initialize framework
    ab_framework = ABTestFramework()
    
    # Create an experiment
    experiment_id = ab_framework.create_experiment(
        name="svd_vs_nmf_ctr",
        description="Compare SVD and NMF algorithms for CTR improvement",
        control_algorithm="svd",
        treatment_algorithm="nmf",
        metric="ctr",
        min_effect_size=0.02,
        statistical_power=0.95,
        significance_level=0.05
    )
    
    print(f"Created experiment: {experiment_id}")
    
    # Run the experiment
    result = ab_framework.run_experiment(experiment_id, duration_days=14)
    
    # Print results
    print("\n" + "="*50)
    print("A/B TEST RESULTS")
    print("="*50)
    print(f"Experiment: {result.experiment_id}")
    print(f"Duration: {(result.end_time - result.start_time).days} days")
    print(f"Control sample size: {result.control_size:,}")
    print(f"Treatment sample size: {result.treatment_size:,}")
    print(f"Control CTR: {result.control_mean:.4f}")
    print(f"Treatment CTR: {result.treatment_mean:.4f}")
    print(f"Lift: {result.lift_percentage:.1f}%")
    print(f"Effect size (Cohen's d): {result.effect_size:.3f}")
    print(f"P-value: {result.p_value:.6f}")
    print(f"Statistical power: {result.statistical_power:.3f}")
    print(f"95% Confidence interval: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")
    print(f"Statistically significant: {result.is_significant}")
    print(f"Recommendation: {result.recommendation}")
    
    return result

if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(demo_ab_testing())
