import dataclasses
from typing import List, Union


@dataclasses.dataclass
class SimulatorConfig:
    model_file: str
    multiple_models_per_cv: bool
    num_pixels: int
    pixel_size: float
    sigma: float
    shift: float
    defocus: List[float]
    snr: List[float]
    amp: float
    b_factor: List[float]


@dataclasses.dataclass
class NeuralNetworkConfig:
    embedding_net: str = "RESNET18"
    out_dim: int = 256
    num_transforms: int = 5
    num_flow_layers: int = 10
    num_flow_nodes: int = 256
    flow_model: str = "NSF"
    theta_scale: float = 1.0
    theta_shift: float = 0.0


@dataclasses.dataclass
class OptimizerConfig:
    optimizer: str = "Adamw"
    lr: float = 0.0003
    weight_decay: float = 0.01


@dataclasses.dataclass
class TrainingConfig:
    start_from_checkpoint: bool = False
    model_checkpoint: Union[str, None] = None
    batch_size: int = 256
    num_epochs: int = 300
    log_interval: int = 10
    save_interval: int = 50
    clip_grad: float = 5.0
    optimizer: OptimizerConfig = OptimizerConfig()
    use_misspecification_loss: bool = False
    misspecification_loss_weight: float = 0.0


@dataclasses.dataclass
class Config:
    simulator: SimulatorConfig 
    neural_network: NeuralNetworkConfig = NeuralNetworkConfig()
    training: TrainingConfig = TrainingConfig()
