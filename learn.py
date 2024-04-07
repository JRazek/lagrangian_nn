import torch

import numpy as np

from integrate_motion import lagrangian_path, plot_vector_field

import random

import lagrangian as lag

import sys

device = 'cpu'
model_path = 'model3.pth'

def harmonic_osillator(t):
    omega = 1
    amplitude = 1
    return amplitude * np.cos(omega * t), -amplitude * omega * np.sin(omega * t), -amplitude * omega * omega * np.cos(omega * t)

def l2_loss(q_dot_dot_predicted, q_dot_dot_actual):
    return (q_dot_dot_predicted - q_dot_dot_actual).pow(2)

def generate_dataset(ranget_start, range_end):
    dataset = []
    targets = []
    for t in range(ranget_start, range_end):
        q, q_dot, q_dot_dot = harmonic_osillator(t)
        dataset.append([q, q_dot])
        targets.append(q_dot_dot)

        
    dataset = torch.tensor(dataset, dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    return dataset, targets


def test_model(model, test_dataset, test_targets):
    for i in range(len(test_dataset)):
        q_dot_dot_predicted = lag.q_dot_dot(model, test_dataset[i])
        print('Predicted: ', q_dot_dot_predicted.item(), 'Actual: ', test_targets[i].item())

def plot_loss(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

def plot_test(model, dataset, targets):
    import matplotlib.pyplot as plt
    for i in range(len(dataset)):
        q_dot_dot_predicted = lag.q_dot_dot(model, dataset[i])
        plt.plot(q_dot_dot_predicted.item(), 'ro')
#        plt.plot(i, targets[i].item(), 'bo')
    plt.show()

def train(model, dataset, targets):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    losses = []
    
    for epoch in range(10):
        loss_total = torch.tensor(0.0, device=device, requires_grad=True)

        for i in range(len(dataset)):
            optimizer.zero_grad()
            q_dot_dot_predicted = lag.q_dot_dot(model, dataset[i])

            if q_dot_dot_predicted.isnan().any():
                continue

            loss_i = l2_loss(q_dot_dot_predicted, targets[i])
            loss_i.backward()
            optimizer.step()
            loss_total = loss_total + loss_i
            
        losses.append(loss_total.detach().numpy())
        print('Epoch: ', epoch, 'Loss: ', loss_total.mean().item())

    return losses

def load_model_or_make_new():
    try:
        model = lag.LagrangianNN().to(device)
        model.load_state_dict(torch.load(model_path))
    except:
        model = lag.LagrangianNN().to(device)
    return model

def main() -> int:
    model = load_model_or_make_new()
    
    dataset, targets = generate_dataset(100, 200)
    
    train(model, dataset, targets)

    torch.save(model.state_dict(), model_path)
    
#    plot_loss(losses)

    test_dataset, test_targets = generate_dataset(100, 200)

#    test_model(model, test_dataset, test_targets)

    plot_vector_field(model, lagrangian_path(model, torch.tensor([0.5, 0.5], dtype=torch.float32, device=device, requires_grad=True)))
    
    return 0

if __name__ == '__main__':
    main()
    sys.exit()
