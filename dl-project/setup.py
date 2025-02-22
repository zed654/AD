from setuptools import setup, find_packages

setup(
    name="dl-project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "typer",
        "minio",
        "loguru",
    ],
    entry_points={
        "console_scripts": [
            "dl-extract=data.data_extract.main:main",
            "dl-train=training.main:main",
            "dl-infer=inference.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning Project for Autonomous Driving",
    url="https://github.com/yourusername/dl-project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
