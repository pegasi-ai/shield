from setuptools import setup, find_packages


def get_requirements(requirements_path):
    with open(requirements_path, "r") as file:
        requirements = [
            line.strip() for line in file if line and not line.startswith("#")
        ]
    return requirements


setup(
    name="pegasi_shield_safeguards",
    version="0.3.0",
    packages=find_packages(),
    install_requires=get_requirements("requirements_lock.txt"),
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