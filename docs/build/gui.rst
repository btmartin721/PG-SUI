PG-SUI Desktop GUI
==================

PG-SUI ships with a MacOS desktop app that wraps the ``pg-sui`` CLI. Every control in the GUI maps directly to a CLI flag, so presets, YAML configs, and ``--set`` overrides behave the same way you would expect on the command line.

Install & launch
----------------

.. note::

   The GUI is currently MacOS-only. Linux users should run PG-SUI via the CLI.

.. tip::

   Install PG-SUI into a fresh virtual environment to keep dependencies isolated.

1. Install the Python package with the GUI extras:

   .. code-block:: bash

      pip install "pg-sui[gui]"

2. Install `Node.js <https://nodejs.org>`__ so ``npm`` is available.

3. Fetch the Electron dependencies (one-time setup):

   .. code-block:: bash

      pgsui-gui-setup

4. Start the app:

   .. code-block:: bash

      pgsui-gui

The GUI uses the active Python environment, so it runs against the same PG-SUI version and configuration you use on the CLI. The ``pgsui-gui-setup`` step only needs to be run once (or after updating PG-SUI) to refresh Electron dependencies.

First run walkthrough
---------------------

1. **Pick a working directory**
   This is where outputs will be written (e.g., ``demo_output/``) and where logs are saved if you set a log file.

2. **Provide input**
   Choose the file format, point to your VCF/PHYLIP/STRUCTURE/GENEPOP file, and optionally add a popmap. Set a ``Prefix`` or keep the inferred filename.

3. **Select models and presets**
   Choose a preset (``fast``, ``balanced``, or ``thorough``) and select one or more models from the multi-select list. Enable tuning if you want Optuna-driven hyperparameter search and specify the number of trials.

4. **Apply configs or overrides**
   Supply a YAML config, dump the effective config to a file, or add one-per-line ``--set`` overrides (e.g., ``model.latent_dim=16``) for quick tweaks. Use include-pops, device, batch size, sim strategy, plot format, and other toggles as needed.

5. **Run**
   Click **Start** to launch. Logs stream live and show the exact CLI command that was executed. Use **Stop** to gracefully terminate a run.

Outputs land in the same layout as the CLI: ``<prefix>_output/<Family>/plots/<Model>/`` and ``<prefix>_output/<Family>/metrics/<Model>/``, plus the ``MultiQC`` report.

GUI to CLI Mapping
------------------

The table below details how specific GUI controls translate to command-line arguments.

.. list-table:: Control Mapping
   :header-rows: 1
   :widths: 25 35 40

   * - GUI Control
     - CLI Flag
     - Notes
   * - **Format**
     - ``--format``
     - ``infer``, ``vcf``, ``phylip``, ``structure``, ``genepop``
   * - **Input file**
     - ``--input`` / ``--vcf``
     - Falls back to ``--vcf`` if a VCF file is selected.
   * - **Popmap**
     - ``--popmap``
     - Optional population map file.
   * - **Force popmap**
     - ``--force-popmap``
     - Forces usage of popmap even if file headers differ.
   * - **Preset**
     - ``--preset``
     - ``fast``, ``balanced``, ``thorough``.
   * - **Models**
     - ``--models``
     - Multi-select allowed.
   * - **YAML config**
     - ``--config``
     - Path to a YAML configuration file.
   * - **Dump config**
     - ``--dump-config``
     - Writes effective configuration to a file.
   * - **Include pops**
     - ``--include-pops``
     - Filter specific populations.
   * - **Sim strategy**
     - ``--sim-strategy``
     - See :ref:`simulated_missingness`.
   * - **Tune / Trials**
     - ``--tune`` / ``--tune-n-trials``
     - Enable Optuna and set trial count.
   * - **Overrides**
     - ``--set``
     - One override per line in the GUI maps to repeated flags.
   * - **Dry run**
     - ``--dry-run``
     - Validates inputs/config without training.
   * - **Verbose**
     - ``--verbose``
     - Enables detailed logging.

Troubleshooting & Tips
----------------------

- Run ``pgsui-gui-setup`` after updating the repository to refresh Electron dependencies.
- If the GUI cannot find ``pgsui/cli.py``, ensure PG-SUI is installed in the current environment or set the ``PGSUI_CLI_DEFAULT`` environment variable to the absolute path of ``cli.py``.

.. tip::

   The log panel shows the exact command string executed. Copy this string to reproduce the run in a terminal for debugging or scripting.