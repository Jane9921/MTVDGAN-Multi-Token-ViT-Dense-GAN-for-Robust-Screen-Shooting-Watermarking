import torch.nn as nn
import torch
import kornia
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    matrix: torch.Tensor = torch.eye(
        3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_scaling_matrix(scale: torch.Tensor,
                            center: torch.Tensor) -> torch.Tensor:
    # angle: torch.Tensor = torch.zeros_like(scale)
    angle: torch.Tensor = torch.zeros(scale.shape[0])
    matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix(angle: torch.Tensor,
                             center: torch.Tensor) -> torch.Tensor:
    scale: torch.Tensor = torch.ones((angle.shape[0], 2))
    matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    return matrix


def translate(image, device, d=8):
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]  # destination size
    trans = torch.ones(c, 2)
    for i in range(c):
        dx = random.uniform(-d, d)
        dy = random.uniform(-d, d)

        trans[i, :] = torch.tensor([
            [dx, dy],
        ])
    translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]
    # warping needs data in the shape of BCHW
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)

    # warp the input tensor
    data_warp: torch.Tensor = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)

    # return in the original shape
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp


def rotate(image, device, d=8):
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]  # destination size
    angle = torch.ones(c)
    center = torch.ones(c, 2)
    for i in range(c):
        # scale_factor
        an = random.uniform(-d, d)
        angle[i] = torch.tensor([an])
        # center
        center[i, :] = torch.tensor([[h / 2 - 1, w / 2 - 1], ])
    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(image)

    angle = angle.expand(image.shape[0])
    center = center.expand(image.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    # affine(tensor, scaling_matrix[..., :2, :3], align_corners=align_corners)
    matrix = rotation_matrix[..., :2, :3]
    # warping needs data in the shape of BCHW
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)

    # warp the input tensor
    data_warp: torch.Tensor = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)

    # return in the original shape
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp


def perspective(image, device, d=8):
    # the source points are the region to crop corners
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]  # destination size
    image_size = h
    points_src = torch.ones(c, 4, 2)
    points_dst = torch.ones(c, 4, 2)
    for i in range(c):
        points_src[i, :, :] = torch.tensor([[
            [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
        ]])

        # the destination points are the image vertexes
        # d=8
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right

        points_dst[i, :, :] = torch.tensor([[
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size],
        ]])
        # compute perspective transform
    M: torch.tensor = kornia.geometry.transform.imgwarp.get_perspective_transform(points_src, points_dst).to(device)

    # warp the original image by the found transform
    data_warp: torch.tensor = kornia.geometry.warp_perspective(image.float(), M, dsize=(h, w)).to(device)

    return data_warp


import torch
import numpy as np

def MoireGen(p_size, theta, x_centered, y_centered):
    
    z = torch.zeros((p_size, p_size), dtype=torch.float32)

    x_centered = torch.tensor(x_centered, dtype=torch.float32)
    y_centered = torch.tensor(y_centered, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    for i in range(p_size):
        for j in range(p_size):
            z1 = torch.sin(
                2 * torch.pi * (freq1 * (x_centered * torch.cos(theta) + y_centered * torch.sin(theta)) / p_size))
            z2 = torch.sin(
                2 * torch.pi * (freq2 * (x_centered * torch.cos(-theta) + y_centered * torch.sin(-theta)) / p_size))

            z[i, j] = torch.min(0.5 * (1 + z1 * z2)) 

    M = (z + 1) / 2
    return M


def Light_Distortion(c, embed_image):
    mask = np.zeros((embed_image.shape))
    mask_2d = np.zeros((embed_image.shape[2], embed_image.shape[3]))
    a = 0.7 + np.random.rand(1) * 0.2
    b = 1.1 + np.random.rand(1) * 0.2
    if c == 0:
        direction = np.random.randint(1, 5)
        for i in range(embed_image.shape[2]):
            mask_2d[i, :] = -((b - a) / (mask.shape[2] - 1)) * (i - mask.shape[3]) + a
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = np.rot90(mask_2d, 1)
        elif direction == 3:
            O = np.rot90(mask_2d, 1)
        elif direction == 4:
            O = np.rot90(mask_2d, 1)
        for batch in range(embed_image.shape[0]):
            for channel in range(embed_image.shape[1]):
                mask[batch, channel, :, :] = mask_2d
    else:
        x = np.random.randint(0, mask.shape[2])
        y = np.random.randint(0, mask.shape[3])
        max_len = np.max([np.sqrt(x ** 2 + y ** 2), np.sqrt((x - 255) ** 2 + y ** 2), np.sqrt(x ** 2 + (y - 255) ** 2),
                          np.sqrt((x - 255) ** 2 + (y - 255) ** 2)])
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                mask[:, :, i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b
        O = mask
    return O


def Moire_Distortion(embed_image):
    Z = np.zeros((embed_image.shape))
    for i in range(3):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand(1) * embed_image.shape[2]
        center_y = np.random.rand(1) * embed_image.shape[3]
        M = MoireGen(embed_image.shape[2], theta, center_x, center_y)
        Z[:, i, :, :] = M
    return Z


class ScreenShooting(nn.Module):

    def __init__(self):
        super(ScreenShooting, self).__init__()

    def forward(self, embed_image):
        noised_image = torch.zeros_like(embed_image)
        device = embed_image.device

        noised_image = perspective(embed_image, device, 2)

        c = np.random.randint(0, 2)
        L = Light_Distortion(c, embed_image)

        Z = Moire_Distortion(embed_image) * 2 - 1
        Li = L.copy()
        Mo = Z.copy()
        noised_image = noised_image * torch.from_numpy(Li).to(device) * 0.85 + torch.from_numpy(Mo).to(device) * 0.15

        noised_image = noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size()).to(device)

        return noised_image


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, embed_image):
        output = embed_image
        return output
