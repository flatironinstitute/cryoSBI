import os
import time
import logging
import mrcfile
import torch
from omegaconf import DictConfig
from torchvision import transforms

import cryo_sbi.utils.image_utils as img_utils
import cryo_sbi.utils.classifier_utils as cls_utils


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_file_list(folder: str) -> list[str]:
    """Return a sorted list of .mrc / .mrcs file paths from folder."""
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".mrc", ".mrcs"))]
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
    torch.backends.cudnn.benchmark = True
    ic = cfg.inference

    device = ic.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("inference.device=cuda but no CUDA device available; falling back to CPU.")
        device = "cpu"

    # Default true for backwards compatibility: training-time simulator produces
    # positive-density images, while typical MRCs have negative-density particles.
    invert_contrast = bool(ic.get("invert_contrast", True))
    sign = -1.0 if invert_contrast else 1.0

    particle_paths = get_file_list(ic.folder_with_mrcs)
    logging.info(f"Found {len(particle_paths)} .mrc / .mrcs files in {ic.folder_with_mrcs}.")
    logging.info("Analyzing:\n" + "\n".join(os.path.basename(p) for p in particle_paths))
    if not particle_paths:
        raise FileNotFoundError(f"No .mrc / .mrcs files found in {ic.folder_with_mrcs}")

    # Peek at the first MRC header so the input image size comes from the data
    # itself, not from a config field that could silently desync.
    with mrcfile.open(particle_paths[0], permissive=True, header_only=True) as _mrc:
        input_size = int(_mrc.header.nx)
    logging.info(
        f"Input image size (from MRC header of {os.path.basename(particle_paths[0])}): {input_size}."
    )

    # Track the spatial size as it flows through the pipeline: downsampling
    # changes the tensor size, and any size-parameterized transform that runs
    # afterwards (e.g. WhitenImage's noise-PSD mask) must match.
    pipeline = []
    effective_size = input_size
    if ic.down_sampled_size is not None and 0 < ic.down_sampled_size < input_size:
        pipeline.append(img_utils.FourierDownSample(image_size=input_size, down_sampled_size=ic.down_sampled_size))
        effective_size = ic.down_sampled_size
        logging.info(f"Downsampling enabled: {input_size} -> {ic.down_sampled_size}.")
    else:
        logging.info(
            f"Downsampling disabled (input_size={input_size}, down_sampled_size={ic.down_sampled_size}); "
            f"images will pass through at size {effective_size}."
        )
    if ic.whitening:
        pipeline.append(img_utils.WhitenImage(effective_size))
    pipeline.append(img_utils.NormalizeIndividual())
    transform = transforms.Compose(pipeline)
    logging.info(
        f"Transform pipeline: {[type(t).__name__ for t in pipeline]} "
        f"(effective_size={effective_size})."
    )

    classifier = cls_utils.load_classifier(cfg.train, ic.estimator_weights, device=device)

    num_workers = int(ic.num_workers)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        # persistent_workers and prefetch_factor are only valid with workers.
        loader_kwargs["prefetch_factor"] = ic.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = img_utils.MRCloader(particle_paths, **loader_kwargs)

    results = []
    start_time = time.time()

    with torch.inference_mode():
        for idx, images in loader:
            # Cast to float32 — MRCs are typically float but some come in as
            # int / half / double; the transforms below need a real dtype.
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            if images.shape[0] > ic.max_batch_size:
                logits_list, emb_list = [], []
                for batch in torch.split(images, ic.max_batch_size, dim=0):
                    logits, embeddings = classifier.logits_embedding(sign * transform(batch))
                    logits_list.append(logits)
                    emb_list.append(embeddings)
                results.append((idx, torch.cat(logits_list).cpu(), torch.cat(emb_list).cpu()))
            else:
                logits, embeddings = classifier.logits_embedding(sign * transform(images))
                results.append((idx, logits.cpu(), embeddings.cpu()))

    results.sort(key=lambda x: x[0])
    likelihoods = torch.cat([r[1] for r in results])
    embeddings  = torch.cat([r[2] for r in results])

    os.makedirs(ic.output_dir, exist_ok=True)
    tag = f"_{ic.suffix}" if ic.suffix else ""
    torch.save(likelihoods, os.path.join(ic.output_dir, f"likelihoods{tag}.pt"))
    torch.save(embeddings,  os.path.join(ic.output_dir, f"embeddings{tag}.pt"))
    logging.info(
        f"Inference completed in {time.time() - start_time:.2f}s "
        f"for {likelihoods.shape[0]} images."
    )
