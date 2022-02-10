import numpy as np
from numpy.linalg import inv
from copy import deepcopy
import torch
import math

def get_device_info(x):
    cuda_check = x.is_cuda
    if cuda_check:
        device = "cuda:{}".format(x.get_device())
    else:
        device = 'cpu'
    return device

def skew(w):
    n = w.shape[0]
    device = get_device_info(w)
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1),
                       w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    else:
        zero1 = torch.zeros(n, 1, 1).to(device)
        # zero1 = torch.zeros(n, 1, 1)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
    return W

def exp_so3(Input):
    device = get_device_info(Input)
    n = Input.shape[0]
    if Input.shape == (n, 3, 3):
        W = Input
        w = skew(Input)
    else:
        w = Input
        W = skew(w)

    wnorm_sq = torch.sum(w * w, dim=1)
    wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)

    wnorm = torch.sqrt(wnorm_sq)
    wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)

    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)
    w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
    w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
    w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
    eps = 1e-7

    R = torch.zeros(n, 3, 3).to(device)

    R[wnorm > eps] = torch.cat((torch.cat((cw - ((w0 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat(((w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w1 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat((-(w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w2 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2)),
                               dim=1)[wnorm > eps]

    R[wnorm <= eps] = torch.eye(3).to(device) + W[wnorm < eps] + 1 / 2 * W[wnorm < eps] @ W[wnorm < eps]
    return R

def exp_se3(S):
    device = get_device_info(S)
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3
    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim = (n,1,1)
    wnorm_inv = 1 / wnorm_unsqueezed  # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)

    eps = 1e-014
    W = skew(w)
    P = torch.eye(3, device=device) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    # P = torch.eye(3) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    P[wnorm < eps] = torch.eye(3, device=device)
    # P[wnorm < eps] = torch.eye(3)
    T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4, device=device))], dim=1)
    # T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4))], dim=1)
    T[:, -1, -1] = 1
    return T

def quaternions_to_rotation_matrices_torch(quaternions):
    
    # initialize
    K = quaternions.shape[0]
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1]**2
    yy = quaternions[:, 2]**2
    zz = quaternions[:, 3]**2
    ww = quaternions[:, 0]**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R

def rotation_matrices_to_quaternions_torch(R):
    qr = 0.5 * torch.sqrt(1+torch.einsum('ijj->i', R)).unsqueeze(1)
    qi = 1/(4*qr) * (R[:, 2,1] - R[:, 1,2]).unsqueeze(1)
    qj = 1/(4*qr) * (R[:, 0,2] - R[:, 2,0]).unsqueeze(1)
    qk = 1/(4*qr) * (R[:, 1,0] - R[:, 0,1]).unsqueeze(1)

    return torch.cat([qr, qi, qj, qk], dim=1)

def proj_minus_one_plus_one(x):
    eps = 1e-6
    x = torch.min(x, (1 - eps) * (torch.ones(x.shape).to(x)))
    x = torch.max(x, (-1 + eps) * (torch.ones(x.shape).to(x)))
    return x

def log_SO3(R):
    batch_size = R.shape[0]
    eps = 1e-4
    trace = torch.sum(R[:, range(3), range(3)], dim=1)

    omega = R * torch.zeros(R.shape).to(R)

    theta = torch.acos(proj_minus_one_plus_one((trace - 1) / 2))

    temp = theta.unsqueeze(-1).unsqueeze(-1)

    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

    omega_vector_temp = torch.sqrt(omega_temp[:, range(3), range(3)] + torch.ones(3).to(R))
    A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
    B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
    C = omega_vector_temp[:, 0]
    omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= eps] = skew(omega_vector) * math.pi

    return omega