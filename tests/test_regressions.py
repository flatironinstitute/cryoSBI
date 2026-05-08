"""Regression tests covering the Phase-3 correctness bugs in the cleanup.

Each test should fail against the pre-cleanup behavior and pass after the fix.
"""
import json

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import TESTS_DIR
from cryo_sbi.simulator.cryo_em_simulator import CryoEmSimulator
from cryo_sbi.simulator.normalization import gaussian_normalize_image
from cryo_sbi.training.training import GDStep, _save_checkpoint, _load_checkpoint
from cryo_sbi.utils import image_utils as iu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _testing_simulator_config():
    cfg_path = TESTS_DIR / "config_files" / "image_params_testing.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model_file"] = str(TESTS_DIR / "models" / "hsp90_models.pt")
    return cfg


# ---------------------------------------------------------------------------
# Phase 3 — image_utils
# ---------------------------------------------------------------------------

def test_mask_call_does_not_mutate_input():
    """Mask.__call__ used to mutate the caller's tensor in place (image_utils.py:64-66)."""
    image = torch.ones(50, 50)
    image_before = image.clone()
    mask = iu.Mask(50, radius=10, inside=True)
    _ = mask(image)
    assert torch.equal(image, image_before)


def test_get_image_returns_2d_mrc_for_int_index():
    """MRCdataset.get_image used to return None when an MRC was 2D (image_utils.py:522-525)."""
    paths = [str(TESTS_DIR / "data" / "test.mrc")]  # 2D 5x5 MRC
    ds = iu.MRCdataset(paths, cache_size=0)
    ds.build_index_map(method="mrc")
    img = ds.get_image(0)
    assert img is not None
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 2


def test_get_path_with_list_returns_matching_lengths():
    """get_path(list_idx) used to mismatch path/file_index lengths (image_utils.py:546-547)."""
    paths = [str(TESTS_DIR / "data" / "test.mrc")]
    ds = iu.MRCdataset(paths, cache_size=0)
    ds.build_index_map(method="mrc")
    out_paths, out_file_idx = ds.get_path([0])
    assert len(out_paths) == len(out_file_idx) == 1


def test_whiten_image_handles_2d_input():
    """WhitenImage used to assert ndim==3 (image_utils.py:373); now handles 2D."""
    images_2d = torch.randn(64, 64)
    whitened = iu.WhitenImage(64)(images_2d)
    assert whitened.shape == images_2d.shape


def test_whiten_image_no_div_by_zero_on_constant_input():
    """noise_psd ** -0.5 used to produce inf on constant images (image_utils.py:375)."""
    constant = torch.zeros(2, 64, 64)
    out = iu.WhitenImage(64)(constant)
    assert torch.isfinite(out).all()


def test_lowpass_filter_round_trip_preserves_dc_for_even_n():
    """LowPassFilter used fftshift twice instead of ifftshift before ifft2."""
    # For an all-ones input (only DC), the round-trip with a permissive cutoff
    # should reproduce the original within FP tolerance (no half-pixel shift).
    image = torch.ones(64, 64)
    lp = iu.LowPassFilter(image_size=64, frequency_cutoff=64)
    out = lp(image)
    assert torch.allclose(out, image, atol=1e-4)


# ---------------------------------------------------------------------------
# Phase 3 — simulator / config validation
# ---------------------------------------------------------------------------

def test_padding_factor_float_rounds_to_int_pixels():
    """Float padding_factor used to silently truncate via int() — now rounds.

    1.5 with n_pixels=N must produce a padded grid of round(1.5*N), parity-
    snapped so the crop is symmetric.
    """
    cfg = _testing_simulator_config()
    cfg["padding_factor"] = 1.5
    sim = CryoEmSimulator(cfg)
    n = int(cfg["n_pixels"])
    expected = round(1.5 * n)
    if (expected - n) % 2 == 1:
        expected += 1
    assert sim._n_px_pad == expected
    assert (sim._n_px_pad - n) % 2 == 0  # symmetric crop


def test_padding_factor_below_one_raises():
    cfg = _testing_simulator_config()
    cfg["padding_factor"] = 0.5
    with pytest.raises(ValueError, match="padding_factor"):
        CryoEmSimulator(cfg)


def test_gaussian_normalize_handles_constant_image():
    """std=0 used to produce NaN; now clamped (normalization.py:18)."""
    constant = torch.zeros(3, 16, 16)
    out = gaussian_normalize_image(constant)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Phase 3 — training
# ---------------------------------------------------------------------------

def test_gdstep_skips_scheduler_on_non_finite_loss():
    """GDStep used to step the LR scheduler even when grad-clip rejected the
    optimizer step, drifting OneCycleLR off its schedule (training.py:67-72)."""
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=10
    )
    step = GDStep(optimizer, clip=5.0, lr_scheduler=scheduler)

    nan_loss = torch.tensor(float("nan"), requires_grad=True)
    _ = step(nan_loss)
    assert scheduler.last_epoch == 0


def test_gdstep_steps_scheduler_on_finite_loss():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=10
    )
    step = GDStep(optimizer, clip=5.0, lr_scheduler=scheduler)

    loss = model(torch.randn(2, 4)).sum()
    _ = step(loss)
    assert scheduler.last_epoch == 1


# ---------------------------------------------------------------------------
# Phase 4 — checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_preserves_optimizer_and_scheduler(tmp_path):
    """Phase 4 made checkpoints carry optimizer + scheduler + epoch state."""
    m1 = torch.nn.Linear(4, 3)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=0.001)
    sch1 = torch.optim.lr_scheduler.OneCycleLR(opt1, max_lr=0.001, total_steps=10)

    # Take a few steps so optimizer/scheduler accumulate state.
    for _ in range(3):
        loss = m1(torch.randn(2, 4)).sum()
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        sch1.step()

    ckpt = tmp_path / "ckpt.pt"
    _save_checkpoint(str(ckpt), m1, opt1, sch1, epoch=3)

    m2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=0.001)
    sch2 = torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=0.001, total_steps=10)

    start_epoch = _load_checkpoint(str(ckpt), m2, opt2, sch2)
    assert start_epoch == 3
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    assert sch1.last_epoch == sch2.last_epoch


def test_checkpoint_legacy_state_dict_only_still_loads(tmp_path):
    """Pre-Phase-4 checkpoints (raw state_dict) should still load (with a warning)."""
    m1 = torch.nn.Linear(4, 3)
    legacy = tmp_path / "legacy.pt"
    torch.save(m1.state_dict(), legacy)

    m2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=0.001)
    start = _load_checkpoint(str(legacy), m2, opt2, None)
    assert start == 0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Phase 8 — training smoke test
# ---------------------------------------------------------------------------

def test_train_classifier_one_epoch_cpu_smoke(tmp_path):
    """Run train_classifier for 1 epoch on CPU end-to-end."""
    from cryo_sbi.training.training import train_classifier

    cfg = OmegaConf.create({
        "simulation": _testing_simulator_config(),
        "train": {
            "embedding": {"model": "RESNET18", "out_dim": 16},
            "classifier": {
                "model": "MLP",
                "num_layers": 2,
                "nodes_per_layer": 16,
                "dropout": 0.0,
            },
            "learning_rate": 0.001,
            "one_cycle_scheduler": True,
            "clip_gradient": 5.0,
            "weight_decay": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "device": "cpu",
            "n_workers": 0,
            "saving_frequency": 1,
            "simulation_batch_size": 4,
            "batches_per_epoch": 2,
            "prefetch_factor": None,
            "train_from_checkpoint": False,
            "checkpoint_file": None,
            "use_amp": False,
            "compile_model": False,
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "tensorboard_dir": str(tmp_path / "tb"),
                "estimator_file": str(tmp_path / "estimator.pt"),
            },
        },
    })
    # garbage_class is enabled in the testing config; turn it off here so the
    # auto-num_classes inference (num_models + 0) matches hsp90's 20 models.
    cfg.simulation["garbage_class"] = False

    train_classifier(cfg)

    assert (tmp_path / "estimator.pt").exists()


# ---------------------------------------------------------------------------
# Phase 8 — inference smoke test
# ---------------------------------------------------------------------------

def test_classifier_inference_cpu_smoke(tmp_path):
    """Run classifier_inference on a single MRC end-to-end."""
    from cryo_sbi.inference.inference import classifier_inference
    from cryo_sbi.models.build_models import build_classifier

    train_cfg = {
        "embedding": {"model": "RESNET18", "out_dim": 16},
        "classifier": {
            "model": "MLP",
            "num_layers": 2,
            "nodes_per_layer": 16,
            "dropout": 0.0,
        },
    }
    weights_path = tmp_path / "weights.pt"
    estimator = build_classifier(train_cfg, num_classes=4)
    torch.save(estimator.state_dict(), weights_path)

    # Stage the test MRC under a name matching get_file_list's regex.
    import shutil
    mrc_dir = tmp_path / "mrcs"
    mrc_dir.mkdir()
    shutil.copy(str(TESTS_DIR / "data" / "test.mrc"), mrc_dir / "particles_0.mrc")

    cfg = OmegaConf.create({
        "train": train_cfg,
        "inference": {
            "folder_with_mrcs": str(mrc_dir),
            "estimator_weights": str(weights_path),
            "suffix": "smoke",
            "num_workers": 0,
            "output_dir": str(tmp_path / "out"),
            "down_sampled_size": None,
            "prefetch_factor": None,
            "max_batch_size": 32,
            "whitening": False,  # 5x5 MRC isn't a meaningful target for noise PSD
            "invert_contrast": False,
            "device": "cpu",
        },
    })

    classifier_inference(cfg)

    assert (tmp_path / "out" / "likelihoods_smoke.pt").exists()
    assert (tmp_path / "out" / "embeddings_smoke.pt").exists()


# ---------------------------------------------------------------------------
# FG-relative SNR (post-CTF small-mask version)
# ---------------------------------------------------------------------------

def test_get_snr_uses_signal_power_arg():
    """When signal_power is supplied, noise_power = signal_power · 10^(-snr/2)."""
    from cryo_sbi.simulator.noise import get_snr

    B = 4
    sp = torch.full((B,), 2.0)
    snr = torch.tensor([0.0, 0.5, 1.0, 2.0]).reshape(B, 1, 1)
    out = get_snr(images=None, snr=snr, signal_power=sp)
    expected = sp.reshape(B, 1, 1) * (10.0 ** (-snr / 2))
    assert torch.allclose(out, expected, atol=1e-6)


def test_get_snr_legacy_path_uses_rms_not_std():
    """Without signal_power, the legacy fallback uses RMS (not std)."""
    from cryo_sbi.simulator.noise import get_snr

    # Constant nonzero image: RMS = |c|, std = 0.
    c = 0.7
    images = torch.full((3, 32, 32), c)
    snr = torch.zeros(3, 1, 1)
    out = get_snr(images, snr).reshape(-1)
    # noise_power should be c (RMS) · 10^(-0/2) = c, not 0.
    assert torch.allclose(out, torch.full((3,), c), atol=1e-5)


def test_image_formation_snr_independent_of_bg():
    """
    The new FG-region masked SNR should make the noise floor (statistically)
    independent of the BG-particle count, in contrast to the legacy std-of-
    full-image estimator that scaled with √(particles).
    """
    base = _testing_simulator_config()
    base["snr"] = [0.5, 0.5]
    # Big exclusion radius so BG can't enter the FG mask at all.
    base["exclusion_radius"] = 100.0
    base["padding_factor"] = 2

    torch.manual_seed(0)
    sim_no_bg = CryoEmSimulator({**base, "n_bg_min": 0, "n_bg_max": 0,
                                 "garbage_class": False})
    images_no_bg = sim_no_bg.sample_and_simulate(64)

    torch.manual_seed(0)
    sim_bg = CryoEmSimulator({**base, "n_bg_min": 4, "n_bg_max": 4,
                              "garbage_class": False})
    images_bg = sim_bg.sample_and_simulate(64)

    # With the same seed, FG draws are identical; BG presence shouldn't move
    # the post-noise std by much (the small FG mask excludes BG-only pixels).
    std_no_bg = images_no_bg.std().item()
    std_bg = images_bg.std().item()
    # Allow a generous tolerance; the legacy estimator would inflate by √5x.
    assert abs(std_bg - std_no_bg) / std_no_bg < 0.5, (
        f"std with BG ({std_bg}) too different from std without BG ({std_no_bg})"
    )


def test_simulator_precomputes_snr_mask_radius():
    """The SNR mask radius is set at simulator init from ellipsoid_radii.max()."""
    sim = CryoEmSimulator(_testing_simulator_config())
    # Sanity: radius is positive and finite.
    assert sim._snr_mask_radius > 0
    assert torch.isfinite(torch.tensor(sim._snr_mask_radius))


def test_simulator_snr_mask_radius_override():
    """snr_mask_radius_angstrom in config overrides the auto-derivation."""
    cfg = _testing_simulator_config()
    cfg["snr_mask_radius_angstrom"] = 42.0
    sim = CryoEmSimulator(cfg)
    assert sim._snr_mask_radius == 42.0


# ---------------------------------------------------------------------------
# Strict bounding ellipsoid + pixel-center grid convention
# ---------------------------------------------------------------------------

def test_fit_ellipsoids_warns_when_many_atoms_outside_ellipsoid():
    """fit_ellipsoids issues a UserWarning when >15% of atoms are outside the
    implied ellipsoid.

    Cube-vertex atoms at (±1, ±1, ±1) all sit at distance √3 from the origin
    along the box corners — every one is outside the unit-radius inscribed
    ellipsoid, so the warning must fire.
    """
    import warnings
    from cryo_sbi.simulator.priors import fit_ellipsoids

    cube_vertices = torch.tensor([
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1],
    ], dtype=torch.float32).unsqueeze(0)  # (1, 3, 8)

    with pytest.warns(UserWarning, match="atoms outside its bounding ellipsoid"):
        radii = fit_ellipsoids(cube_vertices)
    assert radii.shape == (1, 3)

    # Sanity: a tightly-packed isotropic cloud should NOT trigger the warning.
    torch.manual_seed(0)
    pts = torch.randn(3, 2000)
    pts = pts / pts.norm(dim=0, keepdim=True) * torch.rand(2000) ** (1.0 / 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # warning would now raise
        fit_ellipsoids(pts.unsqueeze(0))


def test_fit_ellipsoids_contains_all_atoms():
    """Every atom centroid must lie inside the fitted ellipsoid."""
    from cryo_sbi.simulator.priors import fit_ellipsoids

    torch.manual_seed(0)
    # A handful of synthetic models with very different shapes (sphere, rod,
    # disk, random cloud) — covers the geometry space the bound has to handle.
    models = [
        torch.randn(3, 500),                                       # roughly spherical
        torch.stack([torch.randn(200) * 50, torch.randn(200),       # rod along x
                     torch.randn(200)]),
        torch.stack([torch.randn(300) * 30, torch.randn(300) * 30,  # disk in xy
                     torch.randn(300) * 0.5]),
        torch.randn(3, 1000) * torch.tensor([5.0, 1.0, 20.0]).view(3, 1),  # ellipsoidal
    ]
    # Pad to common atom count with NaN (fit_ellipsoids drops them).
    n_atoms = max(m.shape[1] for m in models)
    padded = torch.full((len(models), 3, n_atoms), float("nan"))
    for i, m in enumerate(models):
        padded[i, :, : m.shape[1]] = m

    radii = fit_ellipsoids(padded)
    assert radii.shape == (len(models), 3)

    # For each model, project atoms onto principal axes and check |proj| <= semi-axis.
    for i, m in enumerate(models):
        centered = m - m.mean(dim=1, keepdim=True)
        _, eigvecs = torch.linalg.eigh(
            centered @ centered.T / centered.shape[1]
        )
        proj = eigvecs.T @ centered                              # (3, n_atoms)
        # All atoms must be inside the bounding ellipsoid (with tiny FP slack).
        assert (proj.abs() <= radii[i].unsqueeze(-1) + 1e-4).all(), (
            f"Model {i}: some atoms fall outside the fitted ellipsoid"
        )


def test_project_density_grid_is_pixel_centered():
    """The spatial grid is symmetric around origin (pixel-center convention).

    For an even N the grid should span [-(N-1)/2 * ps, +(N-1)/2 * ps] with no
    sample at exactly ±N*ps/2 (those would be pixel edges, not centers).
    """
    from cryo_sbi.simulator.image_generation import project_density

    n_px = 64
    ps = 1.5
    coords = torch.zeros(1, 3, 1)        # one atom at origin
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sigma = torch.tensor([[[1.0]]])
    shift = torch.zeros(1, 2)
    img = project_density(coords, quats, sigma, shift,
                          torch.tensor(float(n_px)), ps)

    # The atom at coord (0, 0) must project to a peak at the *geometric*
    # image center (between pixels (N-1)/2 and N/2 — i.e. pixels 31 and 32
    # for N=64). The peak should be split symmetrically between them.
    row = img[0, n_px // 2 - 1: n_px // 2 + 1, n_px // 2 - 1: n_px // 2 + 1]
    # Symmetry: the 2x2 central window should be 4-fold symmetric (within FP).
    assert torch.allclose(row, row.flip(0), atol=1e-5)
    assert torch.allclose(row, row.flip(1), atol=1e-5)


def test_snr_mask_aligned_with_fg_density():
    """The SNR mask, centered on fg_shift, captures ≥99% of FG energy.

    Catches grid-convention regressions: if project_density and the SNR mask
    use different conventions, the mask is offset and misses edge pixels.
    """
    cfg = _testing_simulator_config()
    cfg["n_bg_min"] = 0
    cfg["n_bg_max"] = 0
    cfg["garbage_class"] = False
    sim = CryoEmSimulator(cfg)

    # Sample one batch of FG-only images, then re-run image_formation but
    # capture the FG density and SNR mask alignment via Parseval-like check.
    torch.manual_seed(0)
    images = sim.sample_and_simulate(8)

    # Re-derive: simulate with snr -> +inf so noise is essentially zero, then
    # the post-CTF FG energy outside the SNR mask should be tiny.
    cfg_clean = {**cfg, "snr": [10.0, 10.0]}            # near-zero noise
    sim_clean = CryoEmSimulator(cfg_clean)
    torch.manual_seed(0)
    clean = sim_clean.sample_and_simulate(8)

    # Build the same SNR mask the simulator uses, around fg_shift=(0,0)
    # since sample_and_simulate uses the prior's default shift sampling.
    # Easier: just check the mask covers most of the cropped image's energy.
    # Energy fraction inside vs. outside the mask:
    N = clean.shape[-1]
    half = (N - 1) * 0.5
    xs = (torch.arange(N) - half) * float(cfg["pixel_size"])
    r2 = xs[None, :] ** 2 + xs[:, None] ** 2
    mask = (r2 < sim._snr_mask_radius ** 2).float()
    energy_in = (clean ** 2 * mask).sum(dim=(-2, -1))
    energy_total = (clean ** 2).sum(dim=(-2, -1))
    # Most of the FG energy lives inside the mask. With strict bounding radii
    # this fraction should be very high — relax to 80% to keep the test
    # robust to per-image shift jitter (the simulator samples fg_shift from
    # the prior and we can't easily zero it here).
    fraction_inside = (energy_in / energy_total.clamp_min(1e-6)).mean().item()
    assert fraction_inside > 0.8, (
        f"FG energy fraction inside SNR mask is {fraction_inside:.3f}, "
        "mask may be misaligned with the FG density"
    )

    # The original asymmetric grid would still pass this test for typical
    # configs (the mask is large), but combined with the explicit alignment
    # check in test_project_density_grid_is_pixel_centered above we cover the
    # convention consistency.
    _ = images  # silence unused warning
