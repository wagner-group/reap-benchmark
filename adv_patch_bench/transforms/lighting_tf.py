"""Calculate parameters for lighting transform."""

from __future__ import annotations

import kornia.geometry.transform as kornia_tf
import numpy as np
import torch
from sklearn.cluster import KMeans

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms.geometric_tf import get_transform_matrix
from adv_patch_bench.utils.types import BatchImageTensor, BatchMaskTensor


def _run_kmean_single(
    img: np.ndarray,
    k: int,
    keep_channel: bool = True,
    n_init: int = 10,
    max_iter: int = 300,
):
    # NOTE: Should we cluster by 1D data instead?
    kmean = KMeans(
        n_clusters=k, init="k-means++", n_init=n_init, max_iter=max_iter
    )
    img = img.reshape(-1, 1) if not keep_channel else img
    labels = kmean.fit_predict(img)
    return kmean.inertia_, kmean.cluster_centers_, labels


def _run_kmean(img: np.ndarray, keep_channel: bool = True):
    loss_2, centers_2, labels_2 = _run_kmean_single(
        img, 2, keep_channel=keep_channel
    )
    loss_3, centers_3, labels_3 = _run_kmean_single(
        img, 3, keep_channel=keep_channel
    )
    n = img.shape[-1]
    is_2 = _best_k(loss_2, loss_3, n)
    print("2" if is_2 else "3")
    centers = centers_2 if is_2 else centers_3
    labels = labels_2 if is_2 else labels_3
    return centers, labels


def _best_k(loss_2: float, loss_3: float, n: int):
    # k selection is based on the score proposed by
    # https://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf
    alpha_2 = 1 - 3 / (4 * n)
    alpha_3 = alpha_2 + (1 - alpha_2) / 6
    score_2 = loss_2 / alpha_2
    score_3 = loss_3 / (alpha_3 * loss_2)
    return score_2 < score_3


def _find_canonical_kmean(img: np.ndarray):
    centers, labels = _run_kmean(img, keep_channel=True)

    white_idx = centers.sum(-1).argmin()
    black_idx = centers.sum(-1).argmax()
    white = centers[white_idx]
    black = centers[black_idx]
    beta = black
    alpha = white - beta

    canonical = centers[labels]
    return canonical, alpha, beta


def _fit_polynomial(
    real_pixels_by_channel: list[torch.Tensor],
    syn_obj: torch.Tensor | None = None,
    warped_obj_mask: torch.Tensor | None = None,
    transform_mat: torch.Tensor | None = None,
    interp: str = "bilinear",
    poly_degree: int = 1,
    drop_topk: float = 0.0,
) -> torch.Tensor:
    if not isinstance(syn_obj, torch.Tensor):
        raise ValueError("syn_obj must be provided as torch.Tensor.")

    syn_obj: BatchImageTensor = img_util.coerce_rank(syn_obj, 4)
    syn_obj = kornia_tf.warp_perspective(
        syn_obj,
        transform_mat,
        warped_obj_mask.shape[-2:],
        mode=interp,
        padding_mode="zeros",
    )

    coeffs = []
    for channel in range(3):
        syn_pixels = torch.masked_select(syn_obj[:, channel], warped_obj_mask)
        real_pixels = real_pixels_by_channel[channel]

        # Drop some high values to reduce outliers
        num_kept = round((1 - drop_topk) * len(real_pixels))
        diff = (syn_pixels - real_pixels).abs()
        indices = torch.topk(diff, num_kept, largest=False).indices
        syn_pixels, real_pixels = syn_pixels[indices], real_pixels[indices]

        if syn_pixels.sum() == 0:
            poly = torch.zeros(poly_degree + 1)
            poly[-1] = real_pixels.mean()
        else:
            # Fit a polynomial to each channel independently
            poly = np.polyfit(
                syn_pixels.numpy(), real_pixels.numpy(), poly_degree
            )
            poly = torch.from_numpy(poly).float()
        coeffs.append(poly)

    return torch.stack(coeffs, dim=0)


def _simple_percentile(
    real_pixels: np.ndarray, percentile: int = 10
) -> torch.Tensor:
    """Compute relighting transform by matching histogram percentiles.

    Args:
        real_pixels: 1D tensor of pixel values from real images.
        percentile: Percentile of pixels considered as min and max of scaling
            range. Only used when method is "percentile". Defaults to 10.0.
    """
    assert 0 <= percentile <= 100
    real_pixels = real_pixels.reshape(-1)
    percentile = min(percentile, 100 - percentile)
    min_ = np.nanpercentile(real_pixels, percentile)
    max_ = np.nanpercentile(real_pixels, 100 - percentile)
    coeffs = torch.tensor([[max_ - min_, min_]])
    return coeffs


def compute_relight_params(
    img: torch.Tensor,
    method: str | None = "percentile",
    obj_mask: torch.Tensor | None = None,
    transform_mat: torch.Tensor | None = None,
    src_points: np.ndarray | None = None,
    tgt_points: np.ndarray | None = None,
    transform_mode: str = "perspective",
    **relight_kwargs,
) -> torch.Tensor:
    """Compute params of relighting transform.

    Args:
        img: Image as Numpy array.
        method: Method to use for computing the relighting params. Defaults to
            "percentile".

    Returns:
        Relighting transform params, alpha and beta.
    """
    if img.size == 0 or method is None or method == "none":
        return 1.0, 0.0

    img: BatchImageTensor = img_util.coerce_rank(img, 4)
    obj_mask: BatchMaskTensor = img_util.coerce_rank(obj_mask, 4)
    if transform_mat is None:
        transform_mat = get_transform_matrix(
            src=src_points, tgt=tgt_points, transform_mode=transform_mode
        )
    obj_mask = kornia_tf.warp_perspective(
        obj_mask,
        transform_mat,
        img.shape[-2:],
        mode="nearest",
        padding_mode="zeros",
    )

    if method == "percentile":
        obj_mask = obj_mask == 1
        real_pixels = torch.masked_select(img, obj_mask)
        coeffs = _simple_percentile(real_pixels, **relight_kwargs)
    elif method == "kmean":
        # Take top and bottom centers as max and min
        centers, _ = _run_kmean(img, keep_channel=False)
        max_ = centers.max()
        min_ = centers.min()
        coeffs = [max_ - min_, min_]
        raise NotImplementedError(
            "Currently, k-mean method is unused as it does not work well."
        )
    elif method == "polynomial":
        obj_mask = obj_mask[:, 0] == 1
        real_pixels = []
        for channel in range(3):
            real_pixels.append(torch.masked_select(img[:, channel], obj_mask))
        coeffs = _fit_polynomial(
            real_pixels,
            warped_obj_mask=obj_mask,
            transform_mat=transform_mat,
            **relight_kwargs,
        )
    else:
        raise NotImplementedError(f"Method {method} is not implemented!")

    return coeffs
