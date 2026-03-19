================================================
cryoSBI - Simulation-based Inference for Cryo-EM
================================================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |githubactions|
        

.. |githubactions| image:: https://github.com/flatironinstitute/cryoSBI/actions/workflows/python-package.yml/badge.svg?branch=main
    :alt: Testing Status
    :target: https://github.com/flatironinstitute/cryoSBI/actions

Overview
--------
``cryoSBI`` provides simulation-based inference (SBI) tools for cryo-EM image analysis.
The package supports the full workflow from model preparation and synthetic cryo-EM image
generation to classifier training and amortized inference on particle stacks.

Main functionalities
~~~~~~~~~~~~~~~~~~~~

#. **Model preparation from atomic structures** via ``make_torch_models``.
#. **Cryo-EM simulation pipeline** (CTF, defocus/noise sampling, and image generation).
#. **Classifier training** with configurable embedding backbones and classifier heads.
#. **Inference on MRC particles** with batched data loading and configurable preprocessing.

Installing
----------
To install the module you will have to dowload the repository and create a virtual environment with the required dependencies.
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
        --pdb_files path_to_pdb_1.pdb,path_to_pdb_2.pdb,... \
        --output_file path_to_save_models.pt \
        --atom_selection "name CA"

Hyperparameter / configuration files
------------------------------------
Two JSON files control simulation and training. You can use the examples in
``tests/config_files/`` directly as templates:

#. ``tests/config_files/image_params_testing.json``
#. ``tests/config_files/training_params_mlp.json``
#. ``tests/config_files/training_params_proto.json``

Image simulation config
~~~~~~~~~~~~~~~~~~~~~~~
Example fields:

.. code-block:: json

    {
        "N_PIXELS": 64,
        "PIXEL_SIZE": 2.06,
        "SIGMA": [0.5, 5.0],
        "MODEL_FILE": "tests/models/hsp90_models.pt",
        "SHIFT": 20.0,
        "DEFOCUS": [1.5, 3.5],
        "SNR": [0.05, 0.05],
        "AMP": 0.1,
        "B_FACTOR": [1.0, 100.0]
    }

Training config
~~~~~~~~~~~~~~~
Example fields:

.. code-block:: json

    {
        "EMBEDDING": {
            "MODEL": "RESNET18",
            "OUT_DIM": 128
        },
        "CLASSIFIER": {
            "MODEL": "MLP",
            "NUM_CLASSES": 44,
            "NUM_LAYERS": 8,
            "NODES_PER_LAYER": 128,
            "DROPOUT": 0.05
        },
        "LEARNING_RATE": 0.0005,
        "ONE_CYCLE_SCHEDULER": true,
        "CLIP_GRADIENT": 5.0,
        "WEIGHT_DECAY": 0.01,
        "BATCH_SIZE": 128
    }

Training classifier for amortized inference
-------------------------------------
.. code-block:: bash

    train_classifier \
        --image_config_file path_to_simulation_config_file.json \
        --train_config_file path_to_train_config_file.json \
        --epochs 150 \
        --estimator_file path_to_estimator_file.pt \
        --loss_file posterior.loss \
        --n_workers 4 \
        --simulation_batch_size 5120 \
        --train_device cuda

Inference on cryo-EM particles
------------------------------
.. code-block:: bash

    classifier_inference \
        --folder_with_mrcs path_to_folder_with_mrc_files \
        --estimator_config path_to_estimator_config_file.json \
        --estimator_weights path_to_estimator_weights.pt \
        --output_dir path_to_save_inference_results \
        --file_name inference_results.pt \
        --max_batch_size 256 \
        --num_workers 4 \
        --image_size 256 \
        --prefetch_factor 2

Development and testing
-----------------------
The repository CI runs tests with ``pytest``. To run locally:

.. code-block:: bash

    pip install pytest
    pytest tests/
