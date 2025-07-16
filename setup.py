from setuptools import setup, find_packages


def get_requirements(requirements_path):
    with open(requirements_path, "r") as file:
        requirements = [
            line.strip() for line in file if line and not line.startswith("#")
        ]
    return requirements



setup(

    name="pegasi_shield",
    version="0.0.26",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pegasi_shield.output_detectors.equity": ["models/*.pkl"],
    },
    install_requires=get_requirements("requirements.txt"),
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