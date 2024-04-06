import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = 'cuda'

class LagrangianNN(nn.Module):
    def __init__(self):
        super(LagrangianNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
        
    #takes x = (q, q_dot) as input
    #returns lagrangian of a system
    def forward(self, x):
        f = F.sigmoid(self.fc1(x))
        return F.sigmoid(self.fc2(f))

def q_dot_dot(f, x):
    grad = torch.autograd.grad(f(x), x, create_graph=True)
    hessian = torch.autograd.functional.hessian(f, x, create_graph=True).reshape(2, 2)
    ddlddq_dot_dot = hessian[1, 1].reciprocal() * (grad[0][0][0] - hessian[1, 0] * x[0][1])

    return ddlddq_dot_dot

def lagrangian_function(x):
    m = 1
    k = 1
    return 0.5 * m * x[0][1].pow(2) - 0.5 * k * x[0][0].pow(2)

def harmonic_osillator(t):
    omega = 1
    amplitude = 1
    return amplitude * np.cos(omega * t), -amplitude * omega * np.sin(omega * t), -amplitude * omega * omega * np.cos(omega * t)

def l2_loss(q_dot_dot_predicted, q_dot_dot_actual):
    return (q_dot_dot_predicted - q_dot_dot_actual).pow(2)

def generate_dataset():
    dataset = []
    targets = []
    for t in range(100):
        q, q_dot, q_dot_dot = harmonic_osillator(t)
        dataset.append([q, q_dot])
        targets.append(q_dot_dot)

        
    dataset = torch.tensor(dataset, dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.tensor(targets, dtype=torch.float32, device=device, requires_grad=True)
    return dataset, targets


model = LagrangianNN().to(device)

dataset, targets = generate_dataset()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)

for epoch in range(1000):
    loss = torch.tensor([0], dtype=torch.float32, device=device, requires_grad=True)
    for i in range(len(dataset)):
        optimizer.zero_grad()
        q_dot_dot_predicted = q_dot_dot(model, dataset[i])
        loss_i = l2_loss(q_dot_dot_predicted, targets[i])
        loss = loss.add(loss_i)

    loss.backward()
    optimizer.step()
    print('Epoch: ', epoch, 'Loss: ', loss.item())

#q_dot_dot = q_dot_dot(lagrangian_function, dataset[0])

