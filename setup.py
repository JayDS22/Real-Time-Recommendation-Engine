"""
Setup script for Real-Time Recommendation Engine
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="realtime-recommendation-engine",
    version="1.0.0",
    author="Jay Guwalani",
    author_email="jguwalan@umd.edu",
    description="A high-performance recommendation system with sub-100ms latency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JayDS22/realtime-recommendation-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rec-engine-api=src.api.recommendation_api:main",
            "rec-engine-train=src.models.train_models:main",
            "rec-engine-demo=run_demo:main",
            "rec-engine-setup=scripts.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    keywords="recommendation-system machine-learning collaborative-filtering real-time ml-ops",
    project_urls={
        "Bug Reports": "https://github.com/JayDS22/realtime-recommendation-engine/issues",
        "Source": "https://github.com/JayDS22/realtime-recommendation-engine",
        "Documentation": "https://jayds22.github.io/Portfolio/",
    },
)
