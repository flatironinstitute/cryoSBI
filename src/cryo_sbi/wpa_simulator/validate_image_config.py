def check_params(config: dict) -> None:
    """
    Checks if all necessary parameters are provided.

    Args:
        config (dict): Dictionary containing image parameters.

    Returns:
        None
    """

    needed_keys = [
        "N_PIXELS",
        "PIXEL_SIZE",
        "SIGMA",
        "SHIFT",
        "CTF",
        "NOISE",
        "DEFOCUS",
        "SNR",
        "MODEL_FILE",
        "ROTATIONS",
        "RADIUS_MASK",
        "AMP",
        "B_FACTOR",
        "ELECWAVE",
    ]

    for key in needed_keys:
        assert key in config.keys(), f"Please provide a value for {key}"

    return None
