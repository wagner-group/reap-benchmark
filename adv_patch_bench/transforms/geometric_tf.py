"""Methods for extracting shape and keypoints from segmentation masks.

Functions in this file are used by script that generates REAP transform params.
"""

import cv2 as cv
import numpy as np

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


def get_corners(mask):
    # Check that we have a binary mask
    assert mask.ndim == 2
    assert (mask == 1).sum() + (mask == 0).sum() == mask.shape[0] * mask.shape[
        1
    ]

    # Find contour of the object
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Find convex hull to combine multiple contours and/or fix some occlusion
    cat_contours = np.concatenate(contours, axis=0)
    hull = cv.convexHull(cat_contours, returnPoints=True)

    # Fit polygon to remove some annotation errors and get vertices
    vertices = _detect_polygon(hull)

    # vertices: (distance from left edge, distance from top edge)
    return vertices.reshape(-1, 2), hull


def _detect_polygon(contour):
    eps = cv.arcLength(contour, True) * _POLYGON_ERROR
    vertices = cv.approxPolyDP(contour, eps, True)
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
        box = cv.boxPoints((rect[0], rect[1], 0.0))
    else:
        box = cv.boxPoints(rect)
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
