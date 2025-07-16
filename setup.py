from setuptools import setup, find_packages
import os

def get_requirements():
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir())
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        return requirements
    except FileNotFoundError:
        print("requirements.txt not found, returning empty list")
        return []

setup(
    name="pegasi_shield",
    version="0.0.26",
    packages=find_packages(),
    package_data={'': ['requirements.txt']},
    install_requires=get_requirements(),
    author="Pegasi",
    author_email="placeholder@usepegasi.com",
    description="Monitor and autocorrect LLMs responses",
    url="https://pegasi.ai/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
