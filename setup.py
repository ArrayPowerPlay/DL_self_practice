from setuptools import setup, find_packages

setup(
    name='my_nlp_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'requests',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)