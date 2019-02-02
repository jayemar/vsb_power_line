import setuptools

setuptools.setup(
    name="etl",
    version="1.0.0",
    description="DataLoader for ML applications",
    install_requires=[
        'arrow',
        'docopt',
        'joblib',
        'numpy',
        'pandas',
        'progressbar2',
        'PyYAML',
        'scikit-learn',
        'scipy'
    ],
    packages=setuptools.find_packages()
)
