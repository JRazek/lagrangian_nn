import torch
import torch.nn as nn
import torch.nn.functional as F

class LagrangianNN(nn.Module):
    def __init__(self):
        super(LagrangianNN, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, 1)
        
    #takes x = (q, q_dot) as input
    #returns lagrangian of a system
    def forward(self, x):
        y1 = F.softplus(self.fc1(x))
        y2 = F.softplus(self.fc2(y1))
        y3 = F.softplus(self.fc3(y2))
        y4 = F.softplus(self.fc4(y3))
        return y4


def q_dot_dot(f, x):
    grad, = torch.autograd.grad(f(x), x, create_graph=True)

    hessian = torch.autograd.functional.hessian(f, x, create_graph=True).reshape(2, 2)

    ddq_dot_dot = hessian[1, 1].reciprocal() * (grad[0] - hessian[1, 0] * x[1])

    return ddq_dot_dot

