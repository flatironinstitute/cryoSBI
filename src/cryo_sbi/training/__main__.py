import hydra
from omegaconf import DictConfig

from cryo_sbi.utils.conf_schema import register_configs

# Register dataclass schemas with Hydra's ConfigStore so command-line typos
# (e.g. train.epoch=200 instead of train.epochs=200) error at startup.
register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    from cryo_sbi.training.training import train_classifier
    train_classifier(cfg)


if __name__ == "__main__":
    main()
