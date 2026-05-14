"""
Render generated PDFs to images at three OCR noise tiers:
clean, moderate, heavy.

Each tier represents a controlled point along the input quality axis,
calibrated so that downstream Tesseract CER (reported by run_ocr.py)
is interpretable. Tuning happens by editing NOISE_CONFIGS below; the
degradation functions themselves only consume parameters.
"""

import random
from dataclasses import dataclass

import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from config import IMAGE_DIR, N_FORMS, NOISE_LEVELS, PDF_DIR, RANDOM_SEED


# tunable noise parameters
# if after running run_ocr.py the CER for a tier is too low or too high, adjust the values here. Targets we want to hit empirically:
#   clean    CER ~ 0.01 - 0.05
#   moderate CER ~ 0.10 - 0.20
#   heavy    CER ~ 0.20 - 0.40
#   severe   CER ~ 0.40 - 0.65
#   extreme  CER ~ 0.65 - 0.90

@dataclass
class NoiseConfig:
    """Parameters for one noise tier.

    rotation_range : max absolute rotation in degrees (uniform sample).
    blur_radius    : Gaussian blur radius in pixels.
    contrast_mult  : multiply contrast by this (1.0 = no change, <1 = lower).
    n_smudges     : number of gray ellipse smudges to overlay.
    blue_shift     : amount to subtract from the blue channel for paper aging.
    """
    rotation_range: float
    blur_radius: float
    contrast_mult: float
    n_smudges: int
    blue_shift: int


NOISE_CONFIGS = {
    "clean": NoiseConfig(
        rotation_range=0.0, blur_radius=0.0,
        contrast_mult=1.0, n_smudges=0, blue_shift=0,
    ),
    "moderate": NoiseConfig(
        rotation_range=0.5, blur_radius=0.5,
        contrast_mult=0.85, n_smudges=0, blue_shift=0,
    ),
    "heavy": NoiseConfig(
        rotation_range=1.5, blur_radius=1.0,
        contrast_mult=0.70, n_smudges=0, blue_shift=10,
    ),
        "severe": NoiseConfig(
        rotation_range=3.0, blur_radius=1.5,
        contrast_mult=0.55, n_smudges=3, blue_shift=20,
    ),
    "extreme": NoiseConfig(
        rotation_range=5.0, blur_radius=2.5,
        contrast_mult=0.40, n_smudges=8, blue_shift=30,
    ),
}

# image rendering parameters
PDF_DPI = 200                          # higher = sharper but slower
SMUDGE_RADIUS_RANGE = (15, 40)         # px


# degradation pipeline

def apply_rotation(img, cfg):
    """Random rotation within +/- cfg.rotation_range degrees."""
    if cfg.rotation_range <= 0:
        return img
    angle = random.uniform(-cfg.rotation_range, cfg.rotation_range)
    return img.rotate(angle, fillcolor="white")


def apply_blur(img, cfg):
    """Gaussian blur at cfg.blur_radius."""
    if cfg.blur_radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))


def apply_contrast(img, cfg):
    """Reduce contrast (mimics dim lighting)."""
    if cfg.contrast_mult == 1.0:
        return img
    return ImageEnhance.Contrast(img).enhance(cfg.contrast_mult)


def apply_smudges(img, cfg):
    """Overlay gray ellipses to mimic stains / scanner artifacts."""
    if cfg.n_smudges <= 0:
        return img
    draw = ImageDraw.Draw(img)
    for _ in range(cfg.n_smudges):
        x = random.randint(0, img.width)
        y = random.randint(0, img.height)
        r = random.randint(*SMUDGE_RADIUS_RANGE)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(200, 200, 200))
    return img


def apply_blue_shift(img, cfg):
    """Subtract from blue channel to give a paper aging yellow tint."""
    if cfg.blue_shift <= 0:
        return img
    img = img.convert("RGB")
    arr = np.array(img).astype(int)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - cfg.blue_shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# order matters: rotate before blur (so blur softens rotation edges),
# blur before contrast (so contrast loss compounds), smudges and tint
# applied last on top of the degraded base.
DEGRADATION_PIPELINE = [
    apply_rotation,
    apply_blur,
    apply_contrast,
    apply_smudges,
    apply_blue_shift,
]


def degrade(img, level):
    """Apply the full degradation pipeline for the given noise level."""
    cfg = NOISE_CONFIGS[level]
    for step in DEGRADATION_PIPELINE:
        img = step(img, cfg)
    return img


# entry point

def main():
    """Render each PDF at all noise levels and save to data/images/."""
    # seed before any randomness (rotation angles, smudge positions)
    # so reruns produce identical degraded images.
    random.seed(RANDOM_SEED)

    for form_id in range(N_FORMS):
        pdf_path = PDF_DIR / f"form_{form_id:03d}.pdf"
        # convert_from_path returns one image per page; we only have one page
        base_img = convert_from_path(str(pdf_path), dpi=PDF_DPI)[0]

        for level in NOISE_LEVELS:
            # Each tier gets a fresh copy of the base image so the
            # degradations don't compound across tiers
            degraded = degrade(base_img.copy(), level)
            out_path = IMAGE_DIR / level / f"form_{form_id:03d}.png"
            degraded.save(out_path)

    total = N_FORMS * len(NOISE_LEVELS)
    print(f"Rendered {total} images "
          f"({N_FORMS} forms x {len(NOISE_LEVELS)} levels)")


if __name__ == "__main__":
    main()
