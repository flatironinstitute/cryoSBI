===========
cryo_em_SBI
===========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |githubactions|
        

.. |githubactions| image:: https://github.com/DSilva27/cryo_em_SBI/actions/workflows/python-package.yml/badge.svg?branch=iss6
    :alt: Testing Status
    :target: https://github.com/DSilva27/cryo_em_SBI/actions

Collaborators
-------------

David Silva-Sánchez, Lars Dingeldein, Roberto Covino, Pilar Cossio

Dependencies
------------

1. Lampe
2. SciPy
3. NumPy
4. PyTorch
5. json

Installing
----------

Download this repository
~~~~~~~~~~~~~~~~~~~~~~~~

`https://github.com/DSilva27/cryo_em_SBI.git`

Install the module
~~~~~~~~~~~~~~~~~~
.. code:: bash

    python3 -m pip install .

Using the code
--------------

Generating data from command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    generate_training_data 
        --config_file experiments/benchmark_hsp90/image_params_snr01_128.json \
        --num_train_samples 10 \
        --num_val_samples 10 \
        --file_name "test1" \
        --n_workers 24


Train posterior from command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    train_npe_model \
        --image_config_file experiments/benchmark_hsp90/image_params_training.json \
        --train_config_file experiments/benchmark_hsp90/resnet18_encoder.json\
        --epochs 100 \
        --estimator_file experiments/benchmark_hsp90/posterior_hsp90.estimator \
        --loss_file experiments/benchmark_hsp90/posterior_hsp90.loss 
