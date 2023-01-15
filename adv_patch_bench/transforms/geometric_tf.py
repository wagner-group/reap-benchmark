"""Methods for extracting shape and keypoints from segmentation masks.

Functions in this file are used by script that generates REAP transform params.
"""

from __future__ import annotations

import cv2
import kornia.geometry.transform as kornia_tf
import numpy as np
import torch

_POLYGON_ERROR = 0.04
_SHAPE_TO_VERTICES = {
    "circle": ((0, 1, 2, 3),),
    "triangle_inverted": ((0, 1, 2),),
    "triangle": ((0, 1, 2),),
    "rect": ((0, 1, 2, 3),),
    "diamond": ((0, 1, 2, 3),),
    "pentagon": ((0, 2, 3, 4),),
    "octagon": ((0, 2, 4, 6),),
}
_VALID_TRANSFORM_MODE = ("perspective", "translate_scale")


def get_corners(mask):
    # Check that we have a binary mask
    assert mask.ndim == 2
    assert (mask == 1).sum() + (mask == 0).sum() == mask.shape[0] * mask.shape[
        1
    ]

    # Find contour of the object
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # Find convex hull to combine multiple contours and/or fix some occlusion
    cat_contours = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(cat_contours, returnPoints=True)

    # Fit polygon to remove some annotation errors and get vertices
    vertices = _detect_polygon(hull)

    # vertices: (distance from left edge, distance from top edge)
    return vertices.reshape(-1, 2), hull


def _detect_polygon(contour):
    eps = cv2.arcLength(contour, True) * _POLYGON_ERROR
    vertices = cv2.approxPolyDP(contour, eps, True)
    return vertices


def _find_first_vertex(vertices):
    # Find the two left most vertices and select the top one
    left_two = np.argsort(vertices[:, 0])[:2]
    first_vertex = left_two[np.argsort(vertices[left_two, 1])[0]]
    return first_vertex


def _get_box_from_ellipse(rect):
    DEV_RATIO_THRES = 0.1
    assert len(rect) == 3
    # If width and height are close or angle is very large, the rotation may be
    # incorrectly estimated
    mean_size = (rect[1][0] + rect[1][1]) / 2
    dev_ratio = abs(rect[1][0] - mean_size) / mean_size
    if dev_ratio < DEV_RATIO_THRES:
        # Set angle to 0 when width and height are similar
        box = cv2.boxPoints((rect[0], rect[1], 0.0))
    else:
        box = cv2.boxPoints(rect)
    return box


def _sort_polygon_vertices(vertices):
    """
    Sort vertices such that the first one is the top left corner, and the rest
    follows in clockwise order. First, find a point inside the polygon (e.g.,
    mean of all vertices) and sort vertices by the angles.
    """
    # Compute normalized vectors from mean to vertices
    # print('here')
    # print(vertices)
    mean = vertices.mean(0)
    vec = vertices - mean
    # NOTE: y-coordinate has to be inverted because zero starts at the top
    vec[:, 1] *= -1
    vec_len = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)[:, None]
    vec /= vec_len
    # Compute angle from positive x-axis (negative sign is for clockwise)
    # NOTE: numpy.arctan2 takes y and x (not x and y)
    angles = -np.arctan2(vec[:, 1], vec[:, 0])
    # print(angles)
    sorted_idx = np.argsort(angles)
    vertices = vertices[sorted_idx]
    angles = angles[sorted_idx]

    first_idx = _find_first_vertex(vertices)
    # If shape is diamond, find_first_vertex can be ambiguous
    if (
        -np.pi * 5 / 8 < angles[first_idx] < -np.pi * 3 / 8
        and len(vertices) == 4
    ):
        # We want first_idx for diamond to be the left corner
        first_idx = (first_idx - 1) % 4

    first = np.where(sorted_idx == first_idx)[0][0]
    sorted_idx = np.concatenate([sorted_idx[first:], sorted_idx[:first]])
    return vertices[sorted_idx]


def get_box_vertices(vertices, predicted_shape):
    """To apply perspective transform, we need to extract a set of four points
    from `vertices` of the polygon or the circle we identify. There is a
    separate function for each `predicted_shape`.

    Args:
        vertices (np.ndarray): Array of vertices, shape: (num_vertices, 2)
        predicted_shape (str): Shape of the object

    Returns:
        np.ndarray: Array of vertices of a convex quadrilateral used for
            perspective transform
    """
    if predicted_shape == "circle":
        vertices = _get_box_from_ellipse(vertices)

    # print(vertices)
    vertices = _sort_polygon_vertices(vertices)
    # vertices = vertices[::-1]
    # vertices = vertices[1:] + vertices[0]
    vertices = np.roll(vertices, -1, axis=0)

    if predicted_shape in _SHAPE_TO_VERTICES:
        box = vertices[_SHAPE_TO_VERTICES[predicted_shape]]
        assert box.shape == (4, 2) or box.shape == (3, 2)
    else:
        box = vertices
    return box


def _get_side_angle(vertices):
    side = vertices[0] - vertices[1]
    return abs(np.arctan(side[1] / side[0]))


def get_shape_from_vertices(vertices):
    num_vertices = len(vertices)
    vertices = _sort_polygon_vertices(vertices)
    # height = vertices[:, 1].max() - vertices[:, 1].min()
    if num_vertices == 3:
        angle = _get_side_angle(vertices.astype(np.float32))
        if angle < np.pi / 6:
            shape = "triangle_inverted"
        else:
            shape = "triangle"
    elif num_vertices == 4:
        angle = _get_side_angle(vertices.astype(np.float32))
        side1 = vertices[0] - vertices[1]
        side2 = vertices[1] - vertices[2]
        len1 = np.sqrt((side1**2).sum())
        len2 = np.sqrt((side2**2).sum())
        if max(len1, len2) / min(len1, len2) > 1.5 or angle < np.pi / 8:
            shape = "rect"
        else:
            shape = "diamond"
    elif num_vertices == 5:
        shape = "pentagon"
    elif num_vertices == 8:
        shape = "octagon"
    else:
        shape = "other"
    return shape


def get_transform_matrix(
    src: np.ndarray | list[list[float]] | None = None,
    tgt: np.ndarray | list[list[float]] | None = None,
    transform_mode: str = "perspective",
) -> torch.Tensor:
    """Get transformation matrix and parameters.

    Returns:
        Tuple of (Transform function, transformation matrix, target points).
    """
    if transform_mode not in _VALID_TRANSFORM_MODE:
        raise NotImplementedError(
            f"transform_mode {transform_mode} is not implemented. "
            f"Only supports {_VALID_TRANSFORM_MODE}!"
        )
    if not isinstance(src, np.ndarray):
        src = np.array(src, dtype=np.float32)[:, :2]
    if not isinstance(tgt, np.ndarray):
        tgt = np.array(tgt, dtype=np.float32)[:, :2]
    tgt = tgt[: len(src)].copy()
    src = src[: len(tgt)].copy()

    if transform_mode == "translate_scale":
        # Use corners of axis-aligned bounding box for transform
        # (translation and scaling) instead of real corners.
        min_tgt_x = min(tgt[:, 0])
        max_tgt_x = max(tgt[:, 0])
        min_tgt_y = min(tgt[:, 1])
        max_tgt_y = max(tgt[:, 1])
        tgt = np.array(
            [
                [min_tgt_x, min_tgt_y],
                [max_tgt_x, min_tgt_y],
                [max_tgt_x, max_tgt_y],
                [min_tgt_x, max_tgt_y],
            ]
        )

        min_src_x = min(src[:, 0])
        max_src_x = max(src[:, 0])
        min_src_y = min(src[:, 1])
        max_src_y = max(src[:, 1])
        src = np.array(
            [
                [min_src_x, min_src_y],
                [max_src_x, min_src_y],
                [max_src_x, max_src_y],
                [min_src_x, max_src_y],
            ]
        )

    assert src.shape == tgt.shape, (
        f"src and tgt keypoints don't have the same shape ({src.shape} vs "
        f"{tgt.shape})!"
    )

    if len(src) == 3 or len(tgt) == 3:
        # For triangles which have only 3 keypoints
        transform_mat = cv2.getAffineTransform(src, tgt)
        transform_mat = torch.from_numpy(transform_mat).unsqueeze(0).float()
        new_row = torch.tensor([[[0, 0, 1]]])
        transform_mat = torch.cat([transform_mat, new_row], dim=1)
    else:
        # All other signs use perspective transform
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        transform_mat = kornia_tf.get_perspective_transform(src, tgt)
    assert transform_mat.shape == (1, 3, 3)
    return transform_mat
