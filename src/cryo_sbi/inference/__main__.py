from pathlib import Path

import hydra
from omegaconf import DictConfig

_CONF_DIR = str(Path.cwd() / "conf")


@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="inference")
def main(cfg: DictConfig) -> None:
    from cryo_sbi.inference.inference import classifier_inference
    classifier_inference(cfg)


if __name__ == "__main__":
    main()
