def check_image_params(config: dict) -> None:
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
        "RES",
        "SHIFT",
        "DEFOCUS",
        "SNR",
        "MODEL_FILE",
        "AMP",
        "B_FACTOR",
    ]

    for key in needed_keys:
        assert key in config.keys(), f"Please provide a value for {key}"

    return None
