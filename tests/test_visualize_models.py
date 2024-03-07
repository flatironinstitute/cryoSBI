import torch
import pytest
from cryo_sbi.utils.visualize_models import plot_model


def test_plot_model_scatter():
    model = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    plot_model(
        model, method="scatter"
    )  # No assertion, just checking if it runs without errors


def test_plot_model_sphere():
    model = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    plot_model(
        model, method="sphere"
    )  # No assertion, just checking if it runs without errors


def test_plot_model_invalid_model():
    model = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )  # Invalid shape, should have 3 rows
    with pytest.raises(AssertionError):
        plot_model(model, method="scatter")


def test_plot_model_invalid_method():
    model = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.raises(ValueError):
        plot_model(model, method="invalid_method")
