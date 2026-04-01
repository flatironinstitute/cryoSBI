import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from cryo_sbi.training.training import train_classifier
    train_classifier(cfg)


if __name__ == "__main__":
    main()
