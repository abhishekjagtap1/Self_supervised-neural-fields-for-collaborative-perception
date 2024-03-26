import torch

def sphere2xyz(r, theta, phi):
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)
    return torch.stack([r*x, r*y, r*z], axis=-1)

def get_ray_directions_360(i, j, W, H):
    i, j = i.float() + 0.5, j.float() + 0.5
    phi = j * torch.pi / H - torch.pi / 2.
    theta = i * 2. * torch.pi / W + torch.pi
    directions = sphere2xyz(torch.ones_like(theta), theta, phi)
    return directions