from setuptools import setup, find_packages

setup(
    name="rlhf-ppo-pipeline",
    version="1.0.0",
    description="RLHF/PPO Training Pipeline with Performance Profiling",
    author="Your Name",
    author_email="launchpadinspires@gmail.com",
    url="https://github.com/snthomps/rlhf-ppo-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)