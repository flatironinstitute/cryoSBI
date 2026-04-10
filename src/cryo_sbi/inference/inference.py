import os
import time
import logging
import torch
from omegaconf import DictConfig
from torchvision import transforms

import cryo_sbi.utils.image_utils as img_utils
import cryo_sbi.utils.classifier_utils as cls_utils


torch.backends.cudnn.benchmark = True


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_file_list(folder: str) -> list[str]:
    """Return a sorted list of .mrc file paths from folder."""
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mrc")]
    try:
        paths = sorted(paths, key=lambda x: int(os.path.basename(x).split("_")[1]))
    except (ValueError, IndexError):
        logging.warning("Could not sort MRC files numerically; falling back to alphabetical order.")
        paths = sorted(paths)
    return paths


def classifier_inference(cfg: DictConfig) -> None:
    """
    Run classifier inference on a folder of MRC files.

    Args:
        cfg: Hydra DictConfig with cfg.train (model architecture) and cfg.inference.
    """
    setup_logging()
    ic = cfg.inference

    transform = transforms.Compose([
        img_utils.WhitenImage(ic.image_size) if ic.whitening else img_utils.Identity(),
        img_utils.NormalizeIndividual(),
    ])

    particle_paths = get_file_list(ic.folder_with_mrcs)
    logging.info(f"Found {len(particle_paths)} .mrc files.")
    logging.info("Analyzing:\n" + "\n".join(os.path.basename(p) for p in particle_paths))

    classifier = cls_utils.load_classifier(cfg.train, ic.estimator_weights, device=ic.device)

    loader = img_utils.MRCloader(
        particle_paths,
        num_workers=ic.num_workers,
        pin_memory=True,
        prefetch_factor=ic.prefetch_factor,
        persistent_workers=True,
        in_order=False,
    )

    results = []
    start_time = time.time()

    with torch.inference_mode():
        for idx, images in loader:
            images = images.to(ic.device, non_blocking=True)
            if images.shape[0] > ic.max_batch_size:
                logits_list, emb_list = [], []
                for batch in torch.split(images, ic.max_batch_size, dim=0):
                    logits, embeddings = classifier.logits_embedding(-transform(batch))
                    logits_list.append(logits)
                    emb_list.append(embeddings)
                results.append((idx, torch.cat(logits_list).cpu(), torch.cat(emb_list).cpu()))
            else:
                logits, embeddings = classifier.logits_embedding(-transform(images))
                results.append((idx, logits.cpu(), embeddings.cpu()))

    results.sort(key=lambda x: x[0])
    likelihoods = torch.cat([r[1] for r in results])
    embeddings  = torch.cat([r[2] for r in results])

    os.makedirs(ic.output_dir, exist_ok=True)
    torch.save(likelihoods, os.path.join(ic.output_dir, f"likelihoods_{ic.file_name}"))
    torch.save(embeddings,  os.path.join(ic.output_dir, f"embeddings_{ic.file_name}"))
    logging.info(
        f"Inference completed in {time.time() - start_time:.2f}s "
        f"for {likelihoods.shape[0]} images."
    )
