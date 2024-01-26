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

Train posterior from command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    train_npe_model \
        --image_config_file image_config.json \
        --train_config_file train_config.json\
        --epochs 450 \
        --estimator_file ../posterior/test.estimator \
        --loss_file ../posterior/test.loss \
        --n_workers 4 \
        --train_device cuda \
        --saving_freq 20 \
        --simulation_batch_size 2048
