#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "transformers==3.0.2",
    "tensorboardX==2.0",
    "parlai==0.1.20200716",
    "numpy==1.18.5",
]

setup_requirements = []

test_requirements = []

setup(
    author="Mykyta Makarov",
    author_email="evil.unicorn1@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Chatbot agent for parl.ai environments that uses generative adversarial imitation learning",
    entry_points={"console_scripts": ["gail_chatbot=gail_chatbot.cli:main",],},
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="gail_chatbot",
    name="gail_chatbot",
    packages=find_packages(include=["gail_chatbot", "gail_chatbot.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/eublefar/gail_chatbot",
    version="0.1.0",
    zip_safe=False,
)
