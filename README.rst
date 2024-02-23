===========
cryoSBI - Simulation-based Inference for Cryo-EM
===========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |githubactions|
        

.. |githubactions| image:: https://github.com/DSilva27/cryo_em_SBI/actions/workflows/python-package.yml/badge.svg?branch=main
    :alt: Testing Status
    :target: https://github.com/DSilva27/cryo_em_SBI/actions

Dependencies
------------

1. `Lampe <https://lampe.readthedocs.io/en/stable/>`_.
2. `SciPy <https://scipy.org/>`_.
3. `Numpy <https://numpy.org/>`_.
4. `PyTorch <https://pytorch.org/get-started/locally/>`_.
5. json
6. `mrcfile <https://pypi.org/project/mrcfile/>`_.

Installing
----------

Download this repository
~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    git clone `https://github.com/DSilva27/cryo_em_SBI.git`

Navigate to the cloned repository and install the module
~~~~~~~~~~~~~~~~~~
.. code:: bash
    
    cd cryo_em_SBI

.. code:: bash

    pip install .

Simulating cryo-EM particles
-----------------------------
To simulate cryo-EM particles, you can use the CryoEmSimulator class. The class takes in a simulation config file and simulates cryo-EM particles based on the parameters specified in the config file.

.. code:: python

    from cryo_sbi import CryoEmSimulator
    simulator = CryoEmSimulator("path_to_simulation_config_file.json")
    images, parameters = simulator.simulate(num_sim=10, return_parameters=True)

The simulation config file should be a json file with the following structure:

.. code:: json

    {   
        "N_PIXELS": 128,
        "PIXEL_SIZE": 1.5,
        "SIGMA": [0.5, 5.0],
        "MODEL_FILE": "path_to_models/models.pt",
        "SHIFT": 25.0,
        "DEFOCUS": [0.5, 2.0],
        "SNR": [0.001, 0.5],
        "AMP": 0.1,
        "B_FACTOR": [1.0, 100.0] 
    }

Training an amortized posterior model
--------------------------------------
Training of an amortized posterior can be done using the train_npe_model command line utility. The utility takes in an image config file, a train config file, and other training parameters. The utility trains a neural network to approximate the posterior distribution of the parameters given the images.

.. code:: bash

    train_npe_model \
        --image_config_file path_to_simulation_config_file.json \
        --train_config_file path_to_train_config_file.json\
        --epochs 150 \
        --estimator_file posterior.estimator \
        --loss_file posterior.loss \
        --n_workers 4 \
        --simulation_batch_size 5120 \
        --train_device cuda

The training config file should be a json file with the following structure:

.. code:: json

    {
        "EMBEDDING": "RESNET18",
        "OUT_DIM": 256,
        "NUM_TRANSFORM": 5,
        "NUM_HIDDEN_FLOW": 10,
        "HIDDEN_DIM_FLOW": 256,
        "MODEL": "NSF",
        "LEARNING_RATE": 0.0003,
        "CLIP_GRADIENT": 5.0,
        "THETA_SHIFT": 25,
        "THETA_SCALE": 25,
        "BATCH_SIZE": 256
    }   

Inference
---------
Sampling from the posterior distribution can be done using the sample_posterior function in the estimator_utils module. The function takes in an estimator, images, and other parameters and returns samples from the posterior distribution.

.. code:: python

    import cryo_sbi.utils.estimator_utils as est_utils
    samples = est_utils.sample_posterior(
        estimator=posterior,
        images=images,
        num_samples=20000,
        batch_size=100,
        device="cuda",
    )
