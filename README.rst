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

Summary
-------
cryoSBI is a Python module for simulation-based inference in cryo-electron microscopy. The module provides tools for simulating cryo-EM particles, training an amortized posterior model, and sampling from the posterior distribution.
The code is based on the SBI libary `Lampe <https://lampe.readthedocs.io/en/stable/>`_, which is a using Pytorch. 

Installing
----------
To install the module you will have to dowload the repository and create a virtual environment with the required dependencies.
You can create an environment for example with conda using the following command:

.. code:: bash

    conda create -n cryoSBI python=3.10

After creating the virtual environment, you should install the required dependencies and the module.

Dependencies
------------

1. `Lampe <https://lampe.readthedocs.io/en/stable/>`_.
2. `SciPy <https://scipy.org/>`_.
3. `Numpy <https://numpy.org/>`_.
4. `PyTorch <https://pytorch.org/get-started/locally/>`_.
5. json
6. `mrcfile <https://pypi.org/project/mrcfile/>`_.

Download this repository
------------------------
.. code:: bash

    git clone `https://github.com/DSilva27/cryo_em_SBI.git`

Navigate to the cloned repository and install the module
--------------------------------------------------------
.. code:: bash
    
    cd cryo_em_SBI

.. code:: bash

    pip install .

Tutorial
--------
An introduction tutorial can be found at `tutorials/tutorial.ipynb`. In this tutorial, we go through the whole process of making models for cryoSBI, training an amortized posterior, and analyzing the results.
In the following section, I highlighted cryoSBI key features.

Generate model file to simulate cryo-EM particles
-------------------------------------------------
To generate a model file for simulating cryo-EM particles with the simulator provided in this module, you can use the command line tool `models_to_tensor`.
You will need either a set of pdbs which are indexd or a trr trejectory file which contians all models. The tool will generate a model file that can be used to simulate cryo-EM particles.

.. code:: bash

    models_to_tensor \
        --model_file path_to_models/pdb_{}.pdb \
        --output_file path_to_output_file/output.pt \
        --n_pdbs 100

The output file will be a Pytorch tensor with the shape (number of models, 3, number of pseudo atoms).

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

The Pytorch tensor containing the samples will have the shape (number of samples, number of images). In order to visualize the posterior for each image you can use `matplotlib`.
We can quickly generate a histogram with 50 bins with the following piece of code.

.. code:: python

    import matplotlib.pyplot as plt
    idx_image = 0 # posterior for image with index 0
    plt.hist(samples[:, idx_image].flatten(), np.linspace(0, simulator.max_index, 50))

In this case the x-axis is just the index of the structures in increasing order.
