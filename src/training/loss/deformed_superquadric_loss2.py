import torch
from training.functions.lie_torch import quaternions_to_rotation_matrices_torch

# from training.functions.lie_torch import exp_SE3
class DeformedSuperquadricLoss2():
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

    beta = (
        (
        torch.abs(x[:, 0, :]/a1)**(2/e2) 
        + torch.abs(x[:, 1, :]/a2)**(2/e2)
        )**(e2/e1) + torch.abs(x[:, 2, :]/a3)**(2/e1)
    )**(-e1/2)

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