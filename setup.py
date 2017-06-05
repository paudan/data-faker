from setuptools import setup

setup(
    name='data-faker',
    version='0.1',
    url='',
    license='MIT',
    author='Paulius Danenas',
    author_email='danpaulius@gmail.com',
    description='Generate synthetic datasets for Pandas from their definitions',
    packages=['data_faker', 'data_faker.examples'],
    package_dir={'data_faker': 'data_faker',
                 'data_faker.examples': 'data_faker/examples'},
    install_requires=['numpy', 'faker', 'ruamel.yaml', 'pandas'],
    entry_points={
        'console_scripts': [
            'datafaker=generator:main',
        ],
    },
)
