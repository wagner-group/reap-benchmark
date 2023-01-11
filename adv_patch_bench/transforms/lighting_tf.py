"""Calculate parameters for lighting transform."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


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


def _find_min_max(
    img: np.ndarray, method: str = "percentile", percentile: float = 10.0
) -> tuple[float, float]:
    """Find "high" and "low" values of pixels in img.

    High/low values can be defined as some q-percentile pixels in img or based
    on k-mean clustering.

    Args:
        img: Image as Numpy array.
        method: Method of finding high and low values.
        percentile: Percentile used to determine high/low values (only used when
            method is "percentile").

    Returns:
        Tuple of high and low values.
    """
    if img.ndim == 1:
        img = img.reshape(-1, 1)

    assert method in ("percentile", "kmean")

    if method == "percentile":
        assert 0 <= percentile <= 100
        percentile = min(percentile, 100 - percentile)
        min_ = np.nanpercentile(img, percentile)
        max_ = np.nanpercentile(img, 100 - percentile)

    if method == "kmean":
        # Take top and bottom centers as max and min
        centers, _ = _run_kmean(img, keep_channel=False)
        max_ = centers.max()
        min_ = centers.min()
        raise NotImplementedError(
            "Currently, k-mean method is unused as it does not work well."
        )

    return min_, max_


def compute_relight_params(
    img: np.ndarray, method: str | None = "percentile", percentile: float = 10.0
) -> tuple[float, float]:
    """Compute params of relighting transform.

    Args:
        img: Image as Numpy array.
        method: Method to use for computing the relighting params. Defaults to
            percentile.
        percentile: Percentile of pixels considered as min and max of scaling
            range. Only used when method is "percentile". Defaults to 10.0.

    Returns:
        Relighting transform params, alpha and beta.
    """
    if img.size == 0 or method is None or method == "none":
        return 1.0, 0.0
    min_, max_ = _find_min_max(img, method=method, percentile=percentile)
    beta: float = min_
    alpha: float = max_ - beta
    return alpha, beta
