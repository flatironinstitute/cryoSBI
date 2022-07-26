import torch


def check_params(config):

    # Sections
    for section in ["IMAGES", "PREPROCESSING", "SIMULATION", "TRAINING"]:
        assert (
            section in config.keys()
        ), f"Please provide section {section} in config.ini"

    image_params = config["IMAGES"]
    simulation_params = config["SIMULATION"]
    training_params = config["TRAINING"]
    preproc_params = config["PREPROCESSING"]

    # Images
    for key in ["N_PIXELS", "PIXEL_SIZE", "SIGMA"]:
        assert key in image_params.keys(), f"Please provide a value for {key}"

    # Simulation
    for key in ["N_SIMULATIONS", "MODEL_FILE", "DEVICE", "ROTATIONS"]:
        assert key in simulation_params.keys(), f"Please provide a value for {key}"

    if "cuda" in simulation_params["DEVICE"]:
        assert (
            torch.cuda.is_available()
        ), "Your device is cuda but there is no GPU available"

    # Preprocessing
    for key in ["SHIFT", "CTF", "NOISE", "DEFOCUS", "SNR"]:
        assert key in preproc_params.keys(), f"Please provide a value for {key}"

    # Training

    for key in ["MODEL", "HIDDEN_FEATURES", "NUM_TRANSFORMS", "BATCH_SIZE", "DEVICE"]:
        assert key in training_params.keys(), f"Please provide a value for {key}"

    if "POSTERIOR_NAME" not in training_params.keys():
        training_params["POSTERIOR_NAME"] = "posterior.pkl"

    if "cuda" in training_params["DEVICE"]:
        assert (
            torch.cuda.is_available()
        ), "Your device is cuda but there is no GPU available"

    return
