"""
Installable package definition for MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip()
    for line in (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="medixplain",
    version="1.0.0",
    description=(
        "Explainable disease prediction system that combines multi-model "
        "scikit-learn classifiers with a FAISS-based RAG explanation layer."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Mithin Sagar S",
    author_email="mithinsagar@users.noreply.github.com",
    url="https://github.com/mithinsagar/mediXplain-disease-prediction",
    project_urls={
        "Source": "https://github.com/mithinsagar/mediXplain-disease-prediction",
        "Author": "https://github.com/mithinsagar",
    },
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "notebooks", "docs")),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "medixplain-web=app.app:main",
            "medixplain-prepare=ml.prepare_dataset:main",
            "medixplain-train=ml.train_models:main",
            "medixplain-rag=rag.build_faiss_index:main",
        ],
    },
)
