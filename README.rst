================================================
cryoSBI - Simulation-based Inference for Cryo-EM
================================================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |githubactions|


.. |githubactions| image:: https://github.com/DSilva27/cryo_em_SBI/actions/workflows/python-package.yml/badge.svg?branch=main
    :alt: Testing Status
    :target: https://github.com/DSilva27/cryo_em_SBI/actions

Installing
----------
To install the module you will have to download the repository and create a virtual environment with the required dependencies.
You can create an environment for example with conda using the following command:

.. code:: bash

    conda create -n cryoSBI python=3.10

After creating the virtual environment, you should install the required dependencies and the module.

Dependencies
------------

#. `Zuko <https://pypi.org/project/zuko/>`_.
#. `PyTorch <https://pytorch.org/get-started/locally/>`_.
#. `NumPy <https://numpy.org/>`_.
#. `Matplotlib <https://matplotlib.org/>`_.
#. `SciPy <https://scipy.org/>`_.
#. `TorchVision <https://pytorch.org/vision/stable/>`_.
#. `mrcfile <https://pypi.org/project/mrcfile/>`_.
#. `tqdm <https://pypi.org/project/tqdm/>`_.
#. `Hydra-core <https://hydra.cc/>`_ (>= 1.3).
#. `TensorBoard <https://www.tensorflow.org/tensorboard>`_ (>= 2.14).

Download this repository
------------------------
.. code:: bash

    git clone https://github.com/flatironinstitute/cryoSBI.git

Navigate to the cloned repository and install the module
--------------------------------------------------------
.. code:: bash

    cd cryoSBI

.. code:: bash

    pip install .

Generate model file to simulate cryo-EM particles
-------------------------------------------------
.. code-block:: bash

    make_torch_models \
        --pdb_files path_to_pdb_1.pdb path_to_pdb_2.pdb ... \
        --output_file path_to_save_models.pt \
        --atom_selection "name CA"

Training with Hydra
-------------------
Training is configured via `Hydra <https://hydra.cc/>`_ using YAML config files in the ``conf/`` directory.

**Basic usage** (single-particle, MLP classifier):

.. code-block:: bash

    python scripts/train.py

**Override config groups** (use prototype classifier):

.. code-block:: bash

    python scripts/train.py training=prototype

**Override individual parameters on the command line**:

.. code-block:: bash

    python scripts/train.py \
        train.epochs=200 \
        train.device=cuda \
        train.n_workers=4 \
        train.simulation_batch_size=5120 \
        output.estimator_file=outputs/my_model.pt \
        image.MODEL_FILE=/path/to/models.pt \
        training.CLASSIFIER.NUM_CLASSES=44

**Multi-particle mode**:

.. code-block:: bash

    python scripts/train.py \
        image=multi_particle \
        mode.multi_particle=true \
        image.MODEL_FILE=/path/to/models.pt \
        train.device=cuda

**Hyperparameter sweep** (Hydra multirun):

.. code-block:: bash

    python scripts/train.py --multirun \
        training=mlp,prototype \
        train.simulation_batch_size=1024,5120

Config files
~~~~~~~~~~~~
- ``conf/config.yaml`` — main entry point (defaults + output/train settings)
- ``conf/image/single_particle.yaml`` — single-particle imaging parameters
- ``conf/image/multi_particle.yaml`` — multi-particle imaging parameters
- ``conf/training/mlp.yaml`` — MLP classifier training config
- ``conf/training/prototype.yaml`` — Prototype classifier training config

Key image config parameters:

.. list-table::
    :header-rows: 1

    * - Key
      - Description
    * - ``N_PIXELS``
      - Image size in pixels
    * - ``PIXEL_SIZE``
      - Angstrom per pixel
    * - ``MODEL_FILE``
      - Path to ``.pt`` or ``.npy`` model coordinates file
    * - ``DEFOCUS``
      - CTF defocus range ``[min, max]`` in micrometers
    * - ``SNR``
      - Signal-to-noise ratio range ``[min, max]`` (log10 scale)

Resuming from a checkpoint
--------------------------
Full checkpoints (model weights, optimizer state, scheduler state, loss history) are saved to ``outputs/checkpoints/`` every ``saving_frequency`` epochs.

To resume training:

.. code-block:: bash

    python scripts/train.py \
        train.train_from_checkpoint=true \
        train.checkpoint_file=outputs/checkpoints/checkpoint_epoch_20.pt \
        train.epochs=300

The best model (lowest epoch mean loss) is saved automatically as ``<estimator_file>_best.pt``.

Monitoring training with TensorBoard
-------------------------------------
Training metrics are logged to ``outputs/runs/`` by default.

.. code-block:: bash

    tensorboard --logdir outputs/runs

The following scalars are logged:

- ``Loss/batch`` — per-step training loss
- ``Loss/epoch_mean`` — mean loss over the epoch
- ``Accuracy/epoch_mean`` — top-1 classifier accuracy over the epoch
- ``Gradients/norm`` — gradient norm per step (before clipping)
- ``LR/step`` — learning rate per step

Mixed-precision training (AMP)
-------------------------------
Enable mixed-precision training (~1.5–2× speedup on Ampere/Hopper GPUs):

.. code-block:: bash

    python scripts/train.py train.use_amp=true train.device=cuda

Inference on cryo-EM particles
-------------------------------
.. code-block:: bash

    classifier_inference \
        --folder_with_mrcs path_to_folder_with_mrc_files \
        --estimator_config path_to_train_config_file.json \
        --estimator_weights outputs/estimator.pt \
        --output_dir path_to_save_inference_results \
        --file_name inference_results.pt \
        --max_batch_size 256 \
        --num_workers 4 \
        --image_size 256 \
        --prefetch_factor 2

