
from typing import NamedTuple, Tuple

import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa
import torch


class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int


def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw a circle in a numpy array, inplace.
    The center of the circle is at (row, col) and the radius is given by radius.
    The array is assumed to be square.
    Any pixels outside the array are ignored.
    Circle is white (1) on black (0) background, and is anti-aliased.
    """
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]
    return img


def noisy_circle(
    img_size: int, min_radius: float, max_radius: float, noise_level: float
) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle in a numpy array, with normal noise.
    """

    # Create an empty image
    img = np.zeros((img_size, img_size))

    radius = np.random.randint(min_radius, max_radius)

    # x,y coordinates of the center of the circle
    row, col = np.random.randint(img_size, size=2)

    # Draw the circle inplace
    draw_circle(img, row, col, radius)

    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise

    return img, CircleParams(row, col, radius)


def show_circle(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title('Circle')
    plt.show()


def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        return 0
    if d <= abs(r1 - r2):
        return 1 #Problem  here when circle overlap
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    h1 = r1_sq * np.arccos(d1 / r1)
    h2 = d1 * np.sqrt(r1_sq - d1**2)
    h3 = r2_sq * np.arccos(d2 / r2)
    h4 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = h1 + h2 + h3 + h4
    # print(intersection)
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union

import math
import torch

def torch_circle_intersection_area(r1, r2, d):
    """ Calculate the area of intersection of two circles using PyTorch. """
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)
    pi = torch.tensor(math.pi)

    # Case where there is no intersection
    no_intersection = (d >= r1 + r2)

    # Case where one circle is contained within the other
    one_inside_other = (d <= torch.abs(r1 - r2))

    # Intersection area calculation
    alpha = 2 * torch.acos((r1**2 + d**2 - r2**2) / (2 * r1 * d))
    beta = 2 * torch.acos((r2**2 + d**2 - r1**2) / (2 * r2 * d))
    area1 = 0.5 * beta * r2**2 - 0.5 * r2**2 * torch.sin(beta)
    area2 = 0.5 * alpha * r1**2 - 0.5 * r1**2 * torch.sin(alpha)

    intersection_area = area1 + area2

    # Apply conditions
    intersection_area = torch.where(no_intersection, zero, intersection_area)
    # intersection_area = torch.where(one_inside_other, pi * torch.min(r1, r2)**2, intersection_area)
    intersection_area = torch.where(one_inside_other,one,intersection_area) # As per the formula provided originally

    return intersection_area

def torch_circle_iou(circle1, circle2):
    """ Calculate Intersection over Union (IoU) for two circles defined as (x, y, radius) using PyTorch. """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Distance between the centers of the two circles
    d = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Intersection area
    intersection_area = torch_circle_intersection_area(r1, r2, d)
    # print(intersection_area)
    # Union area
    total_area = torch.tensor(math.pi) * (r1**2 + r2**2)
    union_area = total_area - intersection_area

    # IoU
    iou = intersection_area / union_area

    return iou

# Example circles (x, y, radius) as PyTorch tensors
circle1 = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(3.0))
circle2 = (torch.tensor(2.0), torch.tensor(2.0), torch.tensor(2.0))

# Calculate IoU
torch_circle_iou(circle1, circle2)



if __name__ == '__main__':
    cir_a = (0, 0, 50)
    cir_b = (0, 50, 10)

    A = CircleParams(*cir_a)
    B = CircleParams(*cir_b)
    print(iou(A, B))

    # convert cir_a to torch tensor
    torch_cir_a = torch.tensor(cir_a)
    torch_cir_b = torch.tensor(cir_b)
    print(torch_circle_iou(torch_cir_a, torch_cir_b))







