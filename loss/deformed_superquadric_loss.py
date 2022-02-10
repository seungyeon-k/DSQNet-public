import torch
from training.functions.lie_torch import quaternions_to_rotation_matrices_torch

# from training.functions.lie_torch import exp_SE3
class DeformedSuperquadricLoss():
    def __init__(self, device=None, **kargs):
        self.device = device
        self.weight = kargs['weight']

    def loss(self, x, l_gt, output):
        # target : (N_batch, se3 + params)
        # output : (N_batch, 3 + 1, n_pointclouds)

        position = output[:, :3]
        orientation = output[:, 3:7]
        parameters = output[:, 7:]
        volume = torch.prod(parameters[:, :3], dim=1).unsqueeze(1)

        # decompose
        x_position = x[:,:3,:]
        # x_normal = x[:,3:6,:]
        # x_geom = x[:,6,:]

        # # shape parameter
        # l_output = parameters[:, 3:6]
        # loss_param = torch.mean((l_output - l_gt)**2, dim=1)

        rotation = quaternions_to_rotation_matrices_torch(orientation)
        rotation_t = rotation.permute(0,2,1)
        x_transformed = -rotation_t@position.unsqueeze(2) + rotation_t@x_position
        
        # loss = torch.mean(
        #     volume * (dsq_function(x_transformed, parameters)**parameters[:,3:4] - 1) ** 2, 
        #     dim=1
        #     )

        loss_dist = torch.mean(
            dsq_distance(x_transformed, parameters)**2, 
            dim=1
            )

        # orientation loss
        l_output = rotation[:, :3, 2]
        loss_z = torch.norm(torch.cross(l_output, l_gt, dim=1), dim=1)

        # total loss
        loss = torch.mean(loss_dist) + self.weight * torch.mean(loss_z)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('The loss function is inf or nan.')

        return loss

def dsq_function(x, parameters):
    
    # parameters
    a1 = parameters[:,0:1]
    a2 = parameters[:,1:2]
    a3 = parameters[:,2:3]
    e1 = parameters[:,3:4]
    e2 = parameters[:,4:5]
    k = parameters[:,5:6]

    # evaluate function
    f = k / a3 * x[:, 2, :] + 1
    F = (
        torch.abs(x[:, 0, :]/(a1 * f))**(2/e2) 
        + torch.abs(x[:, 1, :]/(a2 * f))**(2/e2)
        )**(e2/e1) + torch.abs(x[:, 2, :]/a3)**(2/e1)

    return F

# def dsq_distance(x, parameters):

#     # parameters
#     a1 = parameters[:,0:1]
#     a2 = parameters[:,1:2]
#     a3 = parameters[:,2:3]
#     e1 = parameters[:,3:4]
#     e2 = parameters[:,4:5]
#     k = parameters[:,5:6]

#     # evaluate function
#     f = k / a3 * x[:, 2, :] + 1
#     eps = 1e-4
#     beta = (
#         (
#         torch.abs(x[:, 0, :]/(torch.abs(a1 * f) + eps))**(2/e2) 
#         + torch.abs(x[:, 1, :]/(torch.abs(a2 * f) + eps))**(2/e2)
#         )**(e2/e1) + torch.abs(x[:, 2, :]/a3)**(2/e1)
#     + eps)**(-e1/2)

#     # evaluate function
#     F = torch.norm(x, dim=1) * (1 - beta)

#     return F    

def dsq_distance(x, parameters):
    
    # parameters
    a1 = parameters[:,0:1]
    a2 = parameters[:,1:2]
    a3 = parameters[:,2:3]
    e1 = parameters[:,3:4]
    e2 = parameters[:,4:5]
    k = parameters[:,5:6]
    b = parameters[:,6:7]
    cos_alpha = parameters[:,7:8]
    sin_alpha = parameters[:,8:9]
    b = b / torch.max(a1, a2)
    # alpha = torch.atan2(sin_alpha, cos_alpha)

    # epsilon
    eps = 1e-4

    # inverse deformation when b' = max(a1, a2) * b
    beta = torch.atan2(x[:, 1, :], x[:, 0, :])
    # R = torch.cos(alpha - beta) * (x[:, 0, :] ** 2 + x[:, 1, :] ** 2) ** (1/2)
    R = (cos_alpha * torch.cos(beta) + sin_alpha * torch.sin(beta)) * (x[:, 0, :] ** 2 + x[:, 1, :] ** 2) ** (1/2)
    r = 1 / b  - (x[:, 2, :] ** 2 + (1 / b - R) ** 2) ** (1/2)
    gamma = torch.atan2(x[:, 2, :], 1 / b - R)
    # X = x[:, 0, :] - torch.cos(alpha) * (R - r)
    # Y = x[:, 1, :] - torch.sin(alpha) * (R - r)
    X = x[:, 0, :] - cos_alpha * (R - r)
    Y = x[:, 1, :] - sin_alpha * (R - r)
    Z = gamma * 1 / b
    f = k / a3 * Z + 1 

    # # inverse deformation when b' = a1 * b
    # beta = torch.atan2(x[:, 1, :], x[:, 0, :])
    # # R = torch.cos(alpha - beta) * (x[:, 0, :] ** 2 + x[:, 1, :] ** 2) ** (1/2)
    # R = (cos_alpha * torch.cos(beta) + sin_alpha * torch.sin(beta)) * (x[:, 0, :] ** 2 + x[:, 1, :] ** 2) ** (1/2)
    # r = a1 / b  - (x[:, 2, :] ** 2 + (a1 / b - R) ** 2) ** (1/2)
    # gamma = torch.atan2(x[:, 2, :], a1 / b - R)
    # # X = x[:, 0, :] - torch.cos(alpha) * (R - r)
    # # Y = x[:, 1, :] - torch.sin(alpha) * (R - r)
    # X = x[:, 0, :] - cos_alpha * (R - r)
    # Y = x[:, 1, :] - sin_alpha * (R - r)
    # Z = gamma * a1 / b
    # f = k / a3 * Z + 1 

    # # inverse deformation when alpha equals 0
    # gamma = torch.atan2(x[:, 2, :], 1 / b - x[:, 0, :])
    # X = 1 / b  - (x[:, 2, :] ** 2 + (1 / b - x[:, 0, :]) ** 2) ** (1/2)
    # Y = x[:, 1, :]
    # Z = gamma / b
    # f = k / a3 * Z + 1 
    # # X = X / (f + eps)
    # # Y = Y / (f + eps)

    # inverse deformation when alpha equals 0 when b' = a1 * b
    # gamma = torch.atan2(x[:, 2, :], a1 / b - x[:, 0, :])
    # X = a1 / b  - (x[:, 2, :] ** 2 + (a1 / b - x[:, 0, :]) ** 2) ** (1/2)
    # Y = x[:, 1, :]
    # Z = gamma * a1 / b
    # f = k / a3 * Z + 1 
    # f[torch.logical_and(f > 0, f < eps)] = eps
    # f[torch.logical_and(f < 0, f > -eps)] = -eps
    # X_ = X / f
    # Y_ = Y / f

    beta = (
        (
        torch.abs(X/(torch.abs(a1 * f) + eps))**(2/e2) 
        + torch.abs(Y/(torch.abs(a2 * f) + eps))**(2/e2)
        )**(e2/e1) + torch.abs(Z/a3)**(2/e1)
    + eps)**(-e1/2)

    # beta = (
    #     (
    #     torch.abs(X_/a1)**(2/e2) 
    #     + torch.abs(Y_/a2)**(2/e2)
    #     )**(e2/e1) + torch.abs(Z/a3)**(2/e1)
    # + eps)**(-e1/2)

    # evaluate function
    # x_total = torch.cat([X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)], dim=1)
    # F = torch.norm(x_total, dim=1) * (1 - beta)
    F = torch.norm(x, dim=1) * (1 - beta)

    return F 