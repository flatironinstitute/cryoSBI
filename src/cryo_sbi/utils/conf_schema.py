from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class SimulationConfig:
    model_file: str = MISSING
    n_pixels: int = 128
    pixel_size: float = 1.5
    sigma: List[float] = field(default_factory=lambda: [1.0, 1.0])
    shift: float = 15.0
    defocus: List[float] = field(default_factory=lambda: [1.0, 1.0])
    snr: List[float] = field(default_factory=lambda: [0.5, 0.5])
    amp: float = 0.1
    voltage_kv: float = 300.0
    b_factor: List[float] = field(default_factory=lambda: [1.0, 100.0])
    n_bg_min: int = 0
    n_bg_max: int = 2
    padding_factor: float = 1.0
    exclusion_radius: float = 0.0
    max_placement_attempts: int = 1000
    garbage_class: bool = False
    min_garbage: int = 4
    max_garbage: int = 10
    # Optional: a fixed quaternion for testing/debugging the orientation prior
    # (used by QuaternionPrior in cryo_sbi.simulator.priors).
    rotations: Optional[List[float]] = None
    # Override (Angstrom) for the SNR mask radius. When None, the simulator
    # uses max(ellipsoid_radii) + sqrt(2)*shift — strict upper bound on the
    # FG particle's image-domain footprint under any orientation and shift.
    snr_mask_radius_angstrom: Optional[float] = None


@dataclass
class EmbeddingConfig:
    model: str = "REGNETY"
    out_dim: int = 128


@dataclass
class ClassifierConfig:
    """
    Classifier kwargs.

    MLP and PROTOTYPE classifiers consume different fields. Unused fields are
    set to None and filtered out by ``build_classifier`` before being passed to
    the constructor — so a PROTOTYPE config doesn't need to set MLP-only fields.

    ``num_classes`` is intentionally absent: it is inferred at runtime from the
    simulator's model tensor (training) or from the saved state_dict (inference).
    """
    model: str = "MLP"
    # MLP-specific
    num_layers: Optional[int] = None
    nodes_per_layer: Optional[int] = None
    dropout: Optional[float] = None
    # PROTOTYPE-specific
    noise_scale: Optional[float] = None


@dataclass
class OutputConfig:
    checkpoint_dir: str = "${hydra:runtime.output_dir}/checkpoints"
    tensorboard_dir: str = "${hydra:runtime.output_dir}/tensorboard_logs"
    estimator_file: str = "${hydra:runtime.output_dir}/estimator.pt"


@dataclass
class TrainConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    learning_rate: float = 5e-4
    one_cycle_scheduler: bool = True
    clip_gradient: Optional[float] = 5.0
    weight_decay: float = 1e-3
    batch_size: int = 128
    epochs: int = 150
    device: str = "cuda"
    n_workers: int = 4
    saving_frequency: int = 20
    simulation_batch_size: int = 1024
    batches_per_epoch: int = 100
    prefetch_factor: Optional[int] = 4
    train_from_checkpoint: bool = False
    checkpoint_file: Optional[str] = None
    use_amp: bool = False
    compile_model: bool = False
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class InferenceConfig:
    folder_with_mrcs: str = MISSING
    estimator_weights: str = MISSING
    file_name: str = "results"
    num_workers: int = 2
    output_dir: str = "."
    image_size: int = 128
    prefetch_factor: Optional[int] = 2
    max_batch_size: int = 32
    whitening: bool = True
    invert_contrast: bool = True
    device: str = "cuda"


@dataclass
class TrainAppConfig:
    """Top-level config for the ``train_classifier`` entry point."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class InferenceAppConfig:
    """Top-level config for the ``classifier_inference`` entry point."""
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def register_configs() -> None:
    """Register Structured Config schemas with Hydra's ConfigStore.

    Each schema is registered under a name matching the YAML it validates, so
    Hydra applies it during config composition. The YAML files keep providing
    default values; the dataclasses provide types. Unknown keys or wrong types
    at the command line error out at startup.

    Idempotent: safe to call from multiple entry points.
    """

    cs = ConfigStore.instance()
    # Schemas are only registered under explicit alias names. Registering them
    # under the same names that @hydra.main looks up ("config", "inference")
    # triggered Hydra's deprecated automatic-schema-matching, which silently
    # bypassed the YAML defaults list and forced dataclass defaults to apply.
    # Users who want validation can opt in with `- train_schema` /
    # `- inference_schema` in their YAML defaults list.
    cs.store(name="train_schema", node=TrainAppConfig)
    cs.store(name="inference_schema", node=InferenceAppConfig)
    # Per-section schemas, referenced from YAML defaults lists if the user
    # wants individual section validation when composing piecewise.
    cs.store(name="simulation_schema", node=SimulationConfig, package="simulation")
    cs.store(name="train_section_schema", node=TrainConfig, package="train")
    cs.store(name="inference_section_schema", node=InferenceConfig, package="inference")
