data-faker
===========

Generate synthetic datasets which can be used directly for research or train models, using YAML specifications. Currently, only Pandas dataframes are supported as output

Installation
------------

The package can be easily from GitHub repository installed using Python's *pip* utility.

Usage
-----

The usage is simply straightforward, identical to scikit's feature selection module:

.. code:: python

    import data_faker as df

    spec_file = 'examples/distributions.yaml'
    output = 'output.csv'
    df.generate(spec_file, output)

A command line tool is also installed during the setup, which allows to generate datasets and serialize them straight from the command line: ::

    datafaker -o output.csv examples/distributions.yaml

or::

    datafaker --output-file output.csv examples/distributions.yaml

Currently the tool supports only serialization to CSV file. However, one can easily serialize the created dataset to other formats, by generating
Pandas dataframe directly using ``generate_pandas`` method, and using internal pandas methods or third-party tools.

Specification
-------------

TBD

Requirements
------------

This tool requires several other Python libraries to function:
- `NumPy <http://www.numpy.org/>`_
- `Pandas <http://pandas.pydata.org/>`_
- `faker <https://pypi.python.org/pypi/Faker>`_
- `ruamel.yaml <https://pypi.python.org/pypi/ruamel.yaml>`_
