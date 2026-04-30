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


``cryoSBI`` trains a classifier on simulated cryo-EM micrographs to identify
which structural model best explains each observed particle. Simulation,
training, and inference are configured via `Hydra <https://hydra.cc/>`_
Structured Configs and dispatched as console scripts.

.. contents::
    :local:
    :depth: 2


Installation
============

Create a Python 3.10+ environment, then install from source:

.. code:: bash

    git clone https://github.com/flatironinstitute/cryoSBI.git
    cd cryoSBI
    pip install .

Dependencies are pulled in by ``pip install``. The main ones are: ``torch``,
``zuko``, ``numpy``, ``scipy``, ``torchvision``, ``mrcfile``, ``hydra-core>=1.3``,
``tensorboard>=2.14``. ``MDAnalysis`` is needed only for the ``make_torch_models``
PDB → tensor entry point and is **not** installed by default — add it with
``pip install MDAnalysis``.

Console scripts installed by ``pip install``:

- ``make_torch_models`` — build a ``.pt`` model file from a list of PDBs.
- ``train_classifier`` — Hydra entry point for training.
- ``classifier_inference`` — Hydra entry point for inference.

The package does **not** ship config files. Each project provides its own
``conf/`` directory; the entry points are pointed at it via ``--config-path``
and ``--config-name``. Example configs live in ``examples/conf/`` — copy them
into your project and edit. Schema validation (Structured Configs) is
attached automatically on the basis of the ``--config-name``, so typos like
``train.epoch=200`` and wrong types like ``train.batch_size=big`` error at
startup whether or not you list the schema in your YAML's ``defaults``.


Quickstart
==========

End-to-end minimal example, assuming you already have a list of PDB files:

.. code-block:: bash

    # 1. Build the model file from PDBs
    make_torch_models \
        --pdb_files structure_1.pdb,structure_2.pdb,structure_3.pdb \
        --output_file models.pt \
        --atom_selection "name CA"

    # 2. Copy the example configs into your project, edit simulation.yaml
    cp -r /path/to/cryoSBI/examples/conf my_project/conf
    # edit my_project/conf/simulation.yaml — set model_file, pixel_size, etc.

    # 3. Train (defaults: 150 epochs, REGNETY embedding, MLP head)
    train_classifier \
        --config-path "$(pwd)/my_project/conf" \
        --config-name config.yaml \
        simulation.model_file="$(pwd)/models.pt"

    # 4. Inference on a folder of MRCs
    classifier_inference \
        --config-path "$(pwd)/my_project/conf" \
        --config-name inference.yaml \
        inference.folder_with_mrcs=/path/to/mrcs \
        inference.estimator_weights=outputs/<date>/<time>/estimator.pt \
        inference.output_dir=results/


Generate a model file from PDB structures
=========================================

.. code-block:: bash

    make_torch_models \
        --pdb_files structure_1.pdb,structure_2.pdb,structure_3.pdb \
        --output_file models.pt \
        --atom_selection "name CA"

The PDB list is **comma-separated** (no spaces). The output ``.pt`` file
contains a ``(num_models, 3, max_atoms)`` tensor padded with ``+inf`` for
short models; downstream code uses ``isfinite`` to mask the padded positions.

The ``--atom_selection`` flag is an MDAnalysis selection string (e.g.
``"name CA"``, ``"protein and not name H*"``, ``"all"``). The selected atoms
are centered on **their own** center of mass, so any selection — not just
``"all"``.


Configuration
=============

All entry points are Hydra apps. They compose a top-level YAML
(``config.yaml`` for training, ``inference.yaml`` for inference) which in
turn pulls in a ``simulation.yaml`` and a ``train.yaml``. Project layout:

.. code-block:: text

    my_project/
        conf/
            config.yaml
            simulation.yaml
            train.yaml
            inference.yaml
        models/
            models.pt
        outputs/             # created by Hydra at run time

The dataclass schemas in ``cryo_sbi.utils.conf_schema`` (``SimulationConfig``,
``TrainConfig``, ``InferenceConfig``) are the source of truth for valid keys
and defaults. When you run ``train_classifier --config-name config``, Hydra
attaches ``TrainAppConfig`` to your YAML automatically — invalid keys or
types from the CLI or YAML raise at startup.

Key ``simulation`` parameters:

.. list-table::
    :header-rows: 1

    * - Key
      - Description
    * - ``simulation.n_pixels``
      - Image side length in pixels.
    * - ``simulation.pixel_size``
      - Angstrom per pixel.
    * - ``simulation.model_file``
      - Path to ``.pt`` model coordinates (required).
    * - ``simulation.shift``
      - Max in-plane shift in Angstrom (uniform on ``[-shift, shift]²``).
    * - ``simulation.defocus``
      - CTF defocus range ``[min, max]`` in micrometers.
    * - ``simulation.snr``
      - SNR range ``[min, max]``; **interpreted on a log10 scale** internally.
    * - ``simulation.sigma``
      - Gaussian width ``[min, max]`` in Angstrom.
    * - ``simulation.amp``
      - Amplitude contrast (scalar; ``low = high`` is allowed).
    * - ``simulation.b_factor``
      - B-factor range ``[min, max]``.
    * - ``simulation.voltage_kv``
      - Microscope voltage in kV (default 300).
    * - ``simulation.padding_factor``
      - Float ≥ 1.0. The padded grid is ``round(n_pixels * padding_factor)``,
        snapped to the same parity as ``n_pixels`` so cropping stays
        symmetric. ``1.0`` disables padding (single-particle mode).
    * - ``simulation.n_bg_min`` / ``simulation.n_bg_max``
      - Background-particle count range. ``n_bg_max=0`` → single-particle.
    * - ``simulation.exclusion_radius``
      - Extra Angstrom buffer added to the bounding-circle test during
        rejection sampling of background-particle positions.
    * - ``simulation.max_placement_attempts``
      - Per-slot rejection-sampling cap. If exhausted the slot is left
        empty (silent — use a generous value if your canvas is crowded).
    * - ``simulation.garbage_class``
      - When true, an extra "garbage" class is added (random pile-up of
        ``min_garbage..max_garbage`` representatives). Enabled →
        ``train.classifier.num_classes`` is auto-derived as
        ``num_models + 1``; the user-supplied value is ignored.
    * - ``simulation.snr_mask_radius_angstrom``
      - Override for the FG SNR-mask radius. ``null`` → automatic from
        ``max(ellipsoid_radii)``. The simulator emits a ``UserWarning`` for
        any model where >15 % of the atoms fall outside the implied PCA
        ellipsoid (single-atom / collinear models also warn separately);
        if you see this, set an explicit override.

Key ``train`` parameters:

.. list-table::
    :header-rows: 1

    * - Key
      - Description
    * - ``train.embedding.model``
      - Embedding net registry key (e.g. ``REGNETY``, ``RESNET18``,
        ``RESNET50``, ``WIDERES50``).
    * - ``train.embedding.out_dim``
      - Embedding dimensionality.
    * - ``train.classifier.model``
      - Classifier head: ``MLP`` or ``PROTOTYPE``.
    * - ``train.classifier.num_classes``
      - Required when ``garbage_class=false``; auto-set to ``num_models+1``
        otherwise.
    * - ``train.epochs``
      - Number of training epochs.
    * - ``train.batch_size``
      - Per optimizer-step batch size.
    * - ``train.simulation_batch_size``
      - Simulator-output batch size (must be a positive multiple of
        ``batch_size`` — checked at startup).
    * - ``train.batches_per_epoch``
      - Number of simulation batches consumed per epoch.
    * - ``train.n_workers`` / ``train.prefetch_factor``
      - DataLoader pool. Workers run the (CPU-only) prior, the main process
        renders on ``train.device``.
    * - ``train.device``
      - ``cuda`` or ``cpu``.
    * - ``train.use_amp``
      - Mixed-precision training (silently a no-op on CPU).
    * - ``train.compile_model``
      - Wrap the classifier in ``torch.compile`` for additional speedup.
    * - ``train.one_cycle_scheduler``
      - Use ``torch.optim.lr_scheduler.OneCycleLR``.
    * - ``train.clip_gradient``
      - L2-norm gradient clip threshold (``null`` to disable).
    * - ``train.train_from_checkpoint`` / ``train.checkpoint_file``
      - Resume a previous run.

Key ``inference`` parameters:

.. list-table::
    :header-rows: 1

    * - Key
      - Description
    * - ``inference.folder_with_mrcs``
      - Directory of ``.mrc`` files (required).
    * - ``inference.estimator_weights``
      - Trained classifier ``.pt`` (required). Accepts both the weights-only
        ``estimator.pt`` and a periodic checkpoint dict.
    * - ``inference.image_size``
      - Pixel side length expected by the model (default 128).
    * - ``inference.max_batch_size``
      - Max images per forward pass.
    * - ``inference.num_workers`` / ``inference.prefetch_factor``
      - DataLoader pool. ``num_workers=0`` → synchronous loading.
    * - ``inference.whitening``
      - Apply noise-PSD whitening before classification (default true; set
        to ``false`` if you don't trust your noise estimate or are working
        with already-preprocessed images).
    * - ``inference.invert_contrast``
      - Flip the sign of input images. Default ``true`` matches the
        convention of training on positive-density simulations and applying
        to negative-stain MRCs. Set to ``false`` if your data already has
        positive density.
    * - ``inference.device``
      - ``cuda`` or ``cpu``.


Training
========

Single GPU, single model:

.. code-block:: bash

    train_classifier \
        --config-path /path/to/project/conf \
        --config-name config.yaml \
        simulation.model_file=/path/to/models.pt

Common overrides on top of the YAML defaults:

.. code-block:: bash

    train_classifier \
        --config-path /path/to/project/conf \
        --config-name config.yaml \
        simulation.model_file=/path/to/models.pt \
        train.epochs=200 \
        train.device=cuda \
        train.n_workers=8 \
        train.simulation_batch_size=5120

Hyperparameter sweep (Hydra multirun):

.. code-block:: bash

    train_classifier --multirun \
        --config-path /path/to/project/conf --config-name config.yaml \
        train.embedding.model=REGNETY,RESNET18 \
        train.simulation_batch_size=1024,5120

Outputs go to ``outputs/<date>/<time>/`` by default (override with
``hydra.run.dir=...``):

- ``checkpoints/checkpoint_epoch_<N>.pt`` — periodic full state (model +
  optimizer + scheduler + epoch + RNG + AMP scaler).
- ``estimator.pt`` — final weights-only file consumed by
  ``classifier_inference``.
- ``tensorboard_logs/`` — TensorBoard scalars.

Resuming from a checkpoint
--------------------------

.. code-block:: bash

    train_classifier \
        --config-path /path/to/project/conf --config-name config.yaml \
        train.train_from_checkpoint=true \
        train.checkpoint_file=outputs/<date>/<time>/checkpoints/checkpoint_epoch_20.pt \
        train.epochs=200

Periodic checkpoints contain the full training state, so resuming continues
exactly where the previous run left off. ``estimator.pt`` is the final
weights-only file consumed by inference. When resuming, set ``train.epochs``
greater than the resumed epoch — otherwise the run refuses to start.

Mixed-precision (AMP)
---------------------

.. code-block:: bash

    train_classifier ... train.use_amp=true train.device=cuda

Loss-scale state is checkpointed; resume preserves the AMP loop exactly.
The setting is silently ignored on CPU.

Compiling the model
-------------------

.. code-block:: bash

    train_classifier ... train.compile_model=true

First batch is slow (warmup); subsequent batches are typically faster.

Monitoring
----------

.. code-block:: bash

    tensorboard --logdir outputs

Scalars: ``Loss/batch``, ``LR/step``, ``Gradients/norm``, ``Loss/epoch``,
``Accuracy/epoch``, ``LR/epoch``, ``Throughput/epoch``.


Inference
=========

.. code-block:: bash

    classifier_inference \
        --config-path /path/to/project/conf \
        --config-name inference.yaml \
        inference.folder_with_mrcs=/path/to/mrc_folder \
        inference.estimator_weights=outputs/<date>/<time>/estimator.pt \
        inference.output_dir=results/

``inference.yaml`` composes ``train.yaml`` so the same architecture overrides
apply (``train.embedding.model``, ``train.classifier.num_classes``, …) when
reconstructing the classifier from weights — these must match the values
used at training time.

Outputs (in ``inference.output_dir``):

- ``likelihoods_<file_name>.pt`` — per-image classifier logits.
- ``embeddings_<file_name>.pt`` — per-image embedding tensors.
- ``image_index_<file_name>.pt`` — global image indices, in the same order.

Pre-classification image transforms — ``whitening`` and ``invert_contrast``
— are config knobs (see the *Configuration* table); both are on by default
to match the typical "negative-stain MRC + simulator-positive density"
pipeline.

Sanity-checking your simulation
===============================

Before launching a long training run, render a small batch and inspect the
images. The hot-path entry point is
``CryoEmSimulator.sample_and_simulate``:

.. code-block:: python

    from cryo_sbi import CryoEmSimulator
    sim = CryoEmSimulator("conf/simulation.yaml", device="cpu")
    images, params = sim.sample_and_simulate(num_sim=8, return_parameters=True)
    # images: (8, n_pixels, n_pixels) float32

Common things to check:

- **SNR mask coverage.** If you see the
  ``fit_ellipsoids: model X has Y% of atoms outside its bounding ellipsoid``
  warning, the auto-derived FG SNR mask under-covers that model. Either
  override with ``simulation.snr_mask_radius_angstrom=<value>`` or trim the
  responsible model.
- **Garbage class look.** With ``garbage_class=true`` a fraction
  ``1/(num_models+1)`` of the rendered images are random pile-ups —
  visualise a few to confirm they look like the kind of "junk" you want
  the network to recognise.
