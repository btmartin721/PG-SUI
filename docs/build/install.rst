Installation
============

.. image:: https://anaconda.org/btmartin721/pg-sui/badges/version.svg
    :target: https://anaconda.org/btmartin721/pg-sui
    :alt: Conda version

.. image:: https://anaconda.org/btmartin721/pg-sui/badges/platforms.svg
    :target: https://anaconda.org/btmartin721/pg-sui
    :alt: Conda platforms

Command-line installation (pip)
-------------------------------

Install PG-SUI from PyPI inside a fresh virtual environment so dependencies stay isolated:

.. code-block:: bash

    python3 -m venv pg-sui-env
    source pg-sui-env/bin/activate

    pip install pg-sui

    # sanity check
    pg-sui --help

PG-SUI pulls in SNPio and all required ML/plotting dependencies. The CLI works anywhere Python 3.11+ is available.

Anaconda installation
----------------------

If you prefer using Anaconda/Miniconda, install PG-SUI from the btmartin721 channel:

.. code-block:: bash

    conda create -n pg-sui-env -c btmartin721 pg-sui
    conda activate pg-sui-env

    # sanity check
    pg-sui --help

MacOS GUI add-on (Electron)
---------------------------

If you have MacOS and want a point-and-click interface, install the optional Electron wrapper. It shells out to the same CLI under the hood, so presets, YAML configs, and overrides behave identically.

.. code-block:: bash

    # install the Python package with GUI extras (FastAPI/uvicorn helper)
    pip install "pg-sui[gui]"

    # one-time setup to fetch the Electron app dependencies
    pgsui-gui-setup

    # launch the desktop app
    pgsui-gui

Node.js (with npm) is required for the Electron app. The GUI respects the active Python environment, making it easy to reuse the same configs you run via ``pg-sui``.

Development Installation
------------------------

If you want to contribute to PG-SUI, install it in development mode so changes are picked up immediately.

.. code-block:: bash

    git clone https://github.com/btmartin721/PG-SUI.git
    cd PG-SUI

    python3 -m venv pg-sui-env
    source pg-sui-env/bin/activate

    pip3 install -e ".[dev]"

For GUI development, append ``[gui]`` to the editable install and rerun ``pgsui-gui-setup`` inside the repo.

.. note::

    If you are using a Jupyter notebook, you will need to restart the kernel after installing PG-SUI in development mode.
