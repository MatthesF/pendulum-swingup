"""A little reward for height; most of it for being up, still and centred."""

import numpy as np

HEIGHT_WEIGHT = 0.05
TOP_ANGLE = np.deg2rad(10)
LINK_SPEED = 0.5
CART_SPEED = 0.5
CART_POSITION = 1.5
RAIL_COST = 1.0


def reward(angles, speeds, cart_position, cart_speed):
    """Between 0 and 1. The link furthest from upright decides."""
    worst = np.max(np.abs(np.arctan2(np.sin(angles), np.cos(angles))), axis=-1)
    height = (1 + np.cos(worst)) / 2
    error = (
        (worst / TOP_ANGLE) ** 4
        + (np.max(np.abs(speeds), axis=-1) / LINK_SPEED) ** 2
        + (cart_speed / CART_SPEED) ** 2
        + (cart_position / CART_POSITION) ** 2
    )
    return HEIGHT_WEIGHT * height + (1 - HEIGHT_WEIGHT) / (1 + error)
