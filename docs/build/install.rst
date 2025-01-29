Installation
============

Standard Installation
----------------------

The best way to install PG-SUI is to just use pip. It will take care of the dependencies for you. We recommend installing PG-SUI in a virtual environment to avoid conflicts with other packages.

.. code-block:: bash

    python3 -m venv pg-sui-env
    source pg-sui-env/bin/activate

    pip3 install pg-sui


It will take care of the dependencies for you.

Anaconda support is coming soon.

Development Installation
------------------------

If you want to contribute to PG-SUI, you can install it in development mode. This will allow you to make changes to the code and see the effects immediately.

.. code-block:: bash

    git clone https://github.com/btmartin721/PG-SUI.git
    cd PG-SUI

    python3 -m venv pg-sui-env
    source pg-sui-env/bin/activate

    pip3 install -e .

This will install PG-SUI in development mode. You can now make changes to the code and see the effects immediately.

.. note::

    If you are using a Jupyter notebook, you will need to restart the kernel after installing PG-SUI in development mode.
