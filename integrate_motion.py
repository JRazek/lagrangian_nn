import lagrangian as lag
import torch

import scipy.integrate

from itertools import product

def differential(model, x) -> torch.Tensor:
    q_dot_dot = lag.q_dot_dot(model, x)

    d_q_dot = q_dot_dot
    d_q = x[1]

    return torch.tensor([d_q, d_q_dot], dtype=torch.float32)


def lagrangian_path(model, x):
    dt = 0.01

    path = []

    for _ in range(1000):
        path.append(x.detach())
        x = x + differential(model, x) * dt

    return torch.stack(path)

def plot_vector_field(model, path=None):
    import matplotlib.pyplot as plt

    q = torch.linspace(-1, 1, 15)
    q_dot = torch.linspace(-1, 1, 15)

    for q_val, q_dot_val in product(q, q_dot):
        x = torch.tensor([q_val, q_dot_val], dtype=torch.float32, requires_grad=True)
        dx = differential(model, x)

        plt.quiver(q_val, q_dot_val, dx[0], dx[1], scale=20)

    plt.xlabel('q')
    plt.ylabel('q_dot')

    if path is not None:
        plt.plot(path[:, 0], path[:, 1], 'r')

    plt.show()
    
