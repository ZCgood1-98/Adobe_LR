#!/usr/bin/env python3
"""
differentiable_renderer/crop_align.py
======================================
Non-differentiable pre-alignment: find the crop region in the original image
that corresponds to the edited image, using SIFT feature matching.

This handles the case where Lightroom applied a crop before colour editing.
The crop region is found once as a preprocessing step, then the cropped
original is fed into the differentiable renderer for colour optimisation.

Only axis-aligned crops are supported (no rotation).
"""

import cv2
import numpy as np
from PIL import Image


def find_crop_box(orig: np.ndarray, edited: np.ndarray,
                  min_inliers: int = 10,
                  ratio_thresh: float = 0.7) -> tuple:
    """
    Find the axis-aligned crop box in *orig* that best matches *edited*.

    Both inputs are HWC uint8 BGR (OpenCV format) or RGB — colour order
    does not matter since we convert to greyscale internally.

    Returns
    -------
    (x, y, w, h) : tuple[int, int, int, int]
        Crop box in *orig*'s pixel coordinates.
        Returns ``None`` if alignment fails.
    """
    g1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim == 3 else orig
    g2 = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY) if edited.ndim == 3 else edited

    # SIFT feature extraction
    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    # KNN matching with Lowe's ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < min_inliers:
        return None

    pts_orig = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_edit = np.float32([kp2[m.trainIdx].pt for m in good])

    # Find homography: maps edited coords -> orig coords
    H, mask = cv2.findHomography(pts_edit, pts_orig, cv2.RANSAC, 5.0)
    if H is None:
        return None

    inliers = mask.ravel().sum()
    if inliers < min_inliers:
        return None

    # Map the four corners of the edited image into orig coordinates
    eh, ew = edited.shape[:2]
    corners_edit = np.float32([[0, 0], [ew, 0], [ew, eh], [0, eh]]).reshape(-1, 1, 2)
    corners_orig = cv2.perspectiveTransform(corners_edit, H).reshape(-1, 2)

    # Extract axis-aligned bounding box
    oh, ow = orig.shape[:2]
    x_min = max(0, int(round(corners_orig[:, 0].min())))
    y_min = max(0, int(round(corners_orig[:, 1].min())))
    x_max = min(ow, int(round(corners_orig[:, 0].max())))
    y_max = min(oh, int(round(corners_orig[:, 1].max())))

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    if crop_w < 10 or crop_h < 10:
        return None

    return (x_min, y_min, crop_w, crop_h)


def needs_crop(orig_size: tuple, edited_size: tuple,
               ratio_tol: float = 0.02) -> bool:
    """
    Check whether aspect ratios differ enough to require a crop alignment.

    Parameters
    ----------
    orig_size, edited_size : (width, height)
    ratio_tol : float
        Tolerance for aspect-ratio comparison.

    Returns ``True`` if the edited image was likely cropped from the original.
    """
    r_orig = orig_size[0] / orig_size[1]
    r_edit = edited_size[0] / edited_size[1]
    return abs(r_orig - r_edit) >= ratio_tol


def align_original_to_edited(orig_path: str, edited_path: str,
                              force_sift: bool = False) -> Image.Image:
    """
    High-level API: given paths to the original and edited images,
    return the original image cropped and resized to match the edited image.

    When aspect ratios differ, SIFT is used to find the crop region.

    When ``force_sift=True`` (recommended for the renderer fit pipeline),
    SIFT is attempted regardless of aspect ratios — this catches sub-pixel
    content shift from upstream dataset cropping, which materially reduces
    misalignment-driven loss "averaging" behaviour in lab_ab / ms_ssim.
    If SIFT cannot find a confident crop on a same-aspect-ratio pair, we
    fall back to a plain resize (the legacy behaviour).

    Returns
    -------
    PIL.Image.Image
        Cropped (and resized) original, same dimensions as the edited image.
    """
    orig_pil = Image.open(orig_path).convert('RGB')
    edited_pil = Image.open(edited_path).convert('RGB')

    ew, eh = edited_pil.size
    ow, oh = orig_pil.size

    aspect_diff = needs_crop((ow, oh), (ew, eh))

    if not aspect_diff and not force_sift:
        # Legacy fast path: same aspect ratio + caller opted out of forced
        # alignment → just resize.
        return orig_pil.resize((ew, eh), Image.LANCZOS)

    # Try SIFT alignment.
    orig_cv = np.array(orig_pil)[:, :, ::-1]   # RGB -> BGR
    edit_cv = np.array(edited_pil)[:, :, ::-1]
    box = find_crop_box(orig_cv, edit_cv)

    if box is not None:
        x, y, w, h = box
        cropped = orig_pil.crop((x, y, x + w, y + h))
    elif aspect_diff:
        # Aspect ratios differ but SIFT failed → centre-crop fallback.
        cropped = _centre_crop_to_ratio(orig_pil, ew / eh)
    else:
        # Same aspect, SIFT failed → plain resize (legacy behaviour).
        return orig_pil.resize((ew, eh), Image.LANCZOS)

    # Resize to match edited dimensions
    return cropped.resize((ew, eh), Image.LANCZOS)


def _centre_crop_to_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """Centre-crop *img* to match *target_ratio* (w/h)."""
    ow, oh = img.size
    current_ratio = ow / oh

    if current_ratio > target_ratio:
        # Too wide — crop width
        new_w = int(oh * target_ratio)
        x0 = (ow - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, oh))
    else:
        # Too tall — crop height
        new_h = int(ow / target_ratio)
        y0 = (oh - new_h) // 2
        return img.crop((0, y0, ow, y0 + new_h))
