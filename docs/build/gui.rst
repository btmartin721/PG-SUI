PG-SUI Desktop GUI
==================

PG-SUI ships with an Electron desktop app that wraps the ``pg-sui`` CLI. Every control in the GUI maps directly to a CLI flag, so presets, YAML configs, and ``--set`` overrides behave the same way you would expect on the command line.

Install & launch
----------------

1. Install the Python package with the GUI extras:

   .. code-block:: bash

      pip install "pg-sui[gui]"

2. Install Node.js (https://nodejs.org) so ``npm`` is available.
3. Fetch the Electron dependencies (one-time):

   .. code-block:: bash

      pgsui-gui-setup

4. Start the app:

   .. code-block:: bash

      pgsui-gui

The GUI uses the active Python environment, so it will run against the same PG-SUI version and configuration you use on the CLI.

First run walkthrough
---------------------

1. **Pick a working directory** -- This is where outputs will be written (e.g., ``demo_output/``) and where logs are saved if you set a log file.
2. **Provide input** -- Choose the file format, point to your VCF/PHYLIP/STRUCTURE/GENEPOP file, and optionally add a popmap. Set a ``Prefix`` or keep the inferred filename.
3. **Select models and presets** -- Choose a preset (``fast``, ``balanced``, or ``thorough``) and select one or more models from the multi-select list. Enable tuning if you want Optuna-driven hyperparameter search and specify the number of trials.
4. **Apply configs or overrides** -- Supply a YAML config, dump the effective config to a file, and add one-per-line ``--set`` overrides (``model.latent_dim=16``) for quick tweaks. Use include-pops, device, batch size, sim strategy, plot format, and other toggles as needed.
5. **Run** -- Click **Start** to launch. Logs stream live and show the exact CLI command that was executed. Use **Stop** to gracefully terminate a run.

Outputs land in the same layout as the CLI: ``<prefix>_output/<Family>/plots/<Model>/`` and ``<prefix>_output/<Family>/metrics/<Model>/`` plus MultiQC if enabled from the CLI.

How GUI controls map to CLI flags
---------------------------------

- **Format** → ``--format`` (``infer``, ``vcf``, ``phylip``, ``structure``, ``genepop``)
- **Input file** → ``--input`` (falls back to ``--vcf`` if you pick a VCF)
- **Popmap** → ``--popmap``; **Force popmap** → ``--force-popmap``
- **Preset** → ``--preset``; **Models** → ``--models`` (multi-select)
- **YAML config** → ``--config``; **Dump config to** → ``--dump-config``
- **Include pops** → ``--include-pops``; **Sim strategy** → ``--sim-strategy`` (see :ref:`simulated_missingness`)
- **Device**/``batch size``/``n_jobs``/``plot format``/``seed`` → matching CLI flags
- **--set overrides** (one per line) → repeated ``--set key=value`` on the CLI
- **Tune**/``Trials`` → ``--tune`` and ``--tune-n-trials``
- **Dry run** → ``--dry-run`` to validate inputs/config without training
- **Log file** → ``--log-file``; **Verbose** → ``--verbose``

Troubleshooting tips
--------------------

- Run ``pgsui-gui-setup`` after updating the repository to refresh Electron dependencies.
- If the GUI cannot find ``pgsui/cli.py``, ensure PG-SUI is installed in the current environment or set ``PGSUI_CLI_DEFAULT`` to the path of ``cli.py``.
- The log panel shows the exact command executed. Copy it to reproduce the run in a terminal or for debugging.
