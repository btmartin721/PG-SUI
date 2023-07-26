Installation
============

Standard Installation
----------------------

The best way to install PG-SUI is to just use pip:

.. code-block:: bash

    pip install pg-sui


It will take care of the dependencies for you.


Manual Installation
--------------------

If you want to manually install PG-SUI, the dependencies are listed below:

+ python >= 3.8
+ pandas
+ numpy==1.24.3
+ scipy
+ matplotlib
+ seaborn
+ plotly
+ kaleido
+ jupyterlab
+ tqdm
+ toytree
+ pyvolve
+ scikit-learn
+ tensorflow >= 2.7
+ keras >= 2.7
+ xgboost
+ scikeras >= 0.6.0
+ snpio


Python versions earlier than 3.8 are not currently supported.  

Installing the Dependencies
---------------------------

The requirements can mostly be installed with conda. The only module that isn't available on conda is sklearn-genetic-opt, which can be installed with pip.

First, let's create a new conda environment to install the PG-SUI dependencies.

.. code-block:: bash

    conda create -n pg-sui python
    conda activate pg-sui

Now let's install the dependencies. Most are available from the default conda channels, so let's do that first. There are some specific versions that need to be installed, so copy and paste the command below.

.. code-block:: bash

    conda install matplotlib seaborn jupyterlab scikit-learn tqdm pandas numpy scipy xgboost tensorflow keras

If you have an Intel processor, then you should also install the ``scikit-learn-intelex`` package. It speeds up computation if you have an intel CPU.

.. code-block:: bash

    conda install scikit-learn-intelex

Now we install ``toytree`` using the conda-forge channel, since it isn't available on the default channel.

.. code-block:: bash

    conda install -c conda-forge toytree

Finally, let's install ``sklearn-genetic-opt``, which does parameter grid searches using a genetic algorithm, and ``scikeras``, which makes the deep learning models compatible with scikit-learn grid searches. These last two are only available through pip.

.. code-block:: bash

    pip install sklearn-genetic-opt[all] scikeras

You must also install SNPio. See the `SNPio documentation <https://snpio.readthedocs.io>`_ for more information.


Installation Troubleshooting
----------------------------

"use_2to3 is invalid" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users running setuptools v58 may encounter this error during the last step of installation, using pip to install sklearn-genetic-opt.

.. code-block:: shell-session

    $ pip install sklearn-genetic-opt[all]

    ERROR: Command errored out with exit status 1:
    command: /Users/tyler/miniforge3/envs/pg-sui/bin/python3.8 -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-install-6y5g_mhs/deap_1d32f65d60a44056bd7031f3aad44571/setup.py'"'"'; __file__='"'"'/private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-install-6y5g_mhs/deap_1d32f65d60a44056bd7031f3aad44571/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base /private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-pip-egg-info-7hg3hcq2
    cwd: /private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-install-6y5g_mhs/deap_1d32f65d60a44056bd7031f3aad44571/
    Complete output (1 lines):
    error in deap setup command: use_2to3 is invalid.

This occurs during the installation of DEAP, one of the dependencies for sklearn-genetic-opt. As a workaround, first downgrade setuptools, and then proceed with the installation as normal.

.. code-block:: bash

    pip install setuptools==57
    pip install sklearn-genetic-opt[all]


Mac ARM architecture
~~~~~~~~~~~~~~~~~~~~

PG-SUI has been tested on the new Mac M1 chips and is working fine, but some changes to the installation process were necessary as of 9-December-21. Installation was successful using the following.

.. code-block:: bash

    # Install Miniforge3 instead of Miniconda3
    # Download: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    bash ~/Downloads/Miniforge3-MacOSX-arm64.sh

    # Close and re-open terminal

    # Create and activate conda environment
    conda create -n pg-sui python

    # Activate environment
    conda activate pg-sui

    # Install packages
    conda install -c conda-forge matplotlib seaborn jupyterlab scikit-learn tqdm pandas numpy scipy xgboost tensorflow keras sklearn-genetic toytree

    # Downgrade setuptools (may or may not be necessary)
    pip install setuptools==57

    # Install sklearn-genetic-opt and mlflow
    pip install sklearn-genetic-opt mlflow


Any other problems we run into testing on the Mac ARM architecture will be adjusted here. Note that the step installing scikit-learn-intelex was skipped here. PG-SUI will automatically detect the CPU architecture you are running, and forgo importing this package (which will only work on Intel processors).


