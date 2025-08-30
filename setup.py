#!/usr/bin/env python3
"""
Setup script for Gemini CLI.

This script provides easy installation and development setup for the
high-performance Gemini CLI tool.
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Get base requirements
base_requirements = read_requirements('requirements.txt')

# Development requirements
dev_requirements = [
    'pytest>=7.4.0',
    'pytest-asyncio>=0.21.0',
    'pytest-benchmark>=4.0.0',
    'pytest-cov>=4.1.0',
    'black>=23.0.0',
    'isort>=5.12.0',
    'flake8>=6.0.0',
    'mypy>=1.5.0',
    'pre-commit>=3.3.0',
]

# Performance requirements
performance_requirements = [
    'uvloop>=0.17.0',
    'httpx>=0.24.0',
    'redis>=4.5.0',
    'pymongo>=4.3.0',
    'lru-dict>=1.1.8',
]

# Monitoring requirements
monitoring_requirements = [
    'prometheus-client>=0.17.0',
    'memory-profiler>=0.61.0',
    'py-spy>=0.3.0',
]

# Benchmarking requirements
benchmarking_requirements = [
    'locust>=2.15.0',
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'scikit-learn>=1.3.0',
]

setup(
    name="gemini-cli",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-Performance Command-Line Interface for Google's Gemini LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gemini-cli",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gemini-cli/issues",
        "Documentation": "https://github.com/yourusername/gemini-cli#readme",
        "Source Code": "https://github.com/yourusername/gemini-cli",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Terminals",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=base_requirements,
    extras_require={
        "dev": dev_requirements,
        "performance": performance_requirements,
        "monitoring": monitoring_requirements,
        "benchmarking": benchmarking_requirements,
        "all": (
            dev_requirements + 
            performance_requirements + 
            monitoring_requirements + 
            benchmarking_requirements
        ),
    },
    entry_points={
        "console_scripts": [
            "gemini-cli=gemini_cli.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "gemini_cli": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "gemini", "ai", "cli", "performance", "optimization", "llm", 
        "google", "generative-ai", "async", "caching", "monitoring"
    ],
    platforms=["any"],
    license="MIT",
    download_url="https://github.com/yourusername/gemini-cli/archive/v1.0.0.tar.gz",
    provides=["gemini_cli"],
    requires_python=">=3.9",
)


def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "develop":
        print("Setting up development environment...")
        
        # Install development dependencies
        os.system(f"{sys.executable} -m pip install -e .[dev]")
        
        # Install pre-commit hooks
        os.system("pre-commit install")
        
        print("Development environment setup complete!")
        print("\nNext steps:")
        print("1. Copy env.example to .env and configure your API key")
        print("2. Run 'gemini-cli --help' to see available commands")
        print("3. Run 'pytest' to run the test suite")
        print("4. Run 'gemini-cli --interactive' to start using the CLI")
        
    else:
        setup()


if __name__ == "__main__":
    main()

