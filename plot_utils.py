import numpy as np
from utils import draw_sphere_marker
import pybullet as p


def plot_points(points, color=(0, 0, 0, 1)):
    marker_ids = []
    for point in points:
        marker_ids.append(plot_point(point, color))
    return marker_ids


def plot_point(point, color=(0, 0, 0, 1)):
    return draw_sphere_marker((point[0], point[1], 0.4), 0.05, color)

def destroy_points(marker_ids):
    for marker_id in marker_ids:
        p.removeBody(marker_id)
    return 1

