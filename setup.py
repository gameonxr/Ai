from setuptools import find_packages, setup

setup(
    name="ai-body-simulator",
    version="0.1.0",
    description="A brain-agnostic humanoid body and physics simulator foundation.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy>=1.24", "PyYAML>=6.0", "matplotlib>=3.7"],
    extras_require={"test": ["pytest>=8.0", "pytest-cov>=5.0"], "mujoco": ["mujoco>=3.0"]},
    entry_points={"console_scripts": ["ai-sim=cli:main"]},
    python_requires=">=3.10",
)
