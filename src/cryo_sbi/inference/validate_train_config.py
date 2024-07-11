def check_train_params(config: dict) -> None:
    """
    Checks if all necessary parameters are provided.

    Args:
        config (dict): Dictionary containing training parameters.

    Returns:
        None
    """

    needed_keys = [
        "EMBEDDING",
        "OUT_DIM",
        "NUM_TRANSFORM",
        "NUM_HIDDEN_FLOW",
        "HIDDEN_DIM_FLOW",
        "MODEL",
        "LEARNING_RATE",
        "CLIP_GRADIENT",
        "BATCH_SIZE",
        "THETA_SHIFT",
        "THETA_SCALE",
        "WEIGHT_DECAY"
    ]

    for key in needed_keys:
        assert key in config.keys(), f"Please provide a value for {key}"

    return
