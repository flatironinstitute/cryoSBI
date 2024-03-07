import matplotlib.pyplot as plt
import numpy as np
import torch


def _scatter_plot_models(model: torch.Tensor, view_angles : tuple = (30, 45), **plot_kwargs: dict) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(*view_angles)

    ax.scatter(*model, **plot_kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def _sphere_plot_models(model: torch.Tensor, radius: float = 4, view_angles : tuple = (30, 45), **plot_kwargs: dict,) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)

    spheres = []
    for x, y, z in zip(model[0], model[1], model[2]):
        spheres.append((x.item(), y.item(), z.item(), radius))

    for idx, sphere in enumerate(spheres):
        x, y, z, r = sphere

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v)) + x
        y = r * np.outer(np.sin(u), np.sin(v)) + y
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z

        ax.plot_surface(x, y, z, **plot_kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_model(model: torch.Tensor, method: str = "scatter", **kwargs) -> None:
    """
    Plot a model from the tensor.

    Args:
        model (torch.Tensor): Model to plot, should be a 2D tensor with shape (3, num_atoms)
        method (str, optional): Method to use for plotting. Defaults to "scatter". Can be "scatter" or "sphere".
                                "scatter" is fast and simple, "sphere" is a proper 3D representation (Take long to render).
        **kwargs: Additional keyword arguments to pass to the plotting function.

    Returns:
        None

    Raises:
        AssertionError: If the model is not a 2D tensor with shape (3, num_atoms).
        ValueError: If the method is not "scatter" or "sphere".

    """

    assert model.ndim == 2, "Model should be 2D tensor"
    assert model.shape[0] == 3, "Model should have 3 rows"

    if method == "scatter":
        _scatter_plot_models(model, **kwargs)

    elif method == "sphere":
        _sphere_plot_models(model, **kwargs)

    else:
        raise ValueError(f"Unknown method {method}. Use 'scatter' or 'sphere'.")
    
