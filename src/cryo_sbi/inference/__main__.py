import hydra
from omegaconf import DictConfig

from cryo_sbi.utils.conf_schema import register_configs

# Register dataclass schemas with Hydra's ConfigStore so command-line typos
# (e.g. inference.image_sze=64) error at startup instead of being ignored.
register_configs()


@hydra.main(version_base=None, config_path=None, config_name="inference")
def main(cfg: DictConfig) -> None:
    from cryo_sbi.inference.inference import classifier_inference
    classifier_inference(cfg)


if __name__ == "__main__":
    main()
