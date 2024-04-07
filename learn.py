import torch

import numpy as np

from integrate_motion import lagrangian_path, plot_vector_field

import random

import lagrangian as lag

import sys

device = 'cpu'
model_path = 'model.pth'

def harmonic_osillator(t, amplitude, omega):
    return amplitude * np.cos(omega * t), -amplitude * omega * np.sin(omega * t), -amplitude * omega**2 * np.cos(omega * t)

def l2_loss(q_dot_dot_predicted, q_dot_dot_actual):
    return (q_dot_dot_predicted - q_dot_dot_actual).pow(2)

def generate_dataset(range_start, range_end, random_amplitude, random_omega):
    dataset = []
    targets = []
    for t in range(range_start, range_end):
        q, q_dot, q_dot_dot = harmonic_osillator(t, random_amplitude, random_omega)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    
    loss_total = torch.tensor(0.0, device=device, requires_grad=True)

    for i in range(len(dataset)):
        optimizer.zero_grad()
        q_dot_dot_predicted = lag.q_dot_dot(model, dataset[i])

        if q_dot_dot_predicted.isnan().any():
            print('Nan')
            continue

        loss_i = l2_loss(q_dot_dot_predicted, targets[i])
        loss_total = loss_total + loss_i

    loss_total.backward()
    optimizer.step()
        
    losses.append(loss_total)

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

    
    mean_losses_epoch = []
    for i in range(100):
        print('Epoch: ', i)


        random_amplitude = random.uniform(0, 1)
        random_omega = random.uniform(0, 1)

        random_n = random.randint(0, 100)

        dataset, targets = generate_dataset(random_n, random_n + 30, random_amplitude, random_omega)
        losses = train(model, dataset, targets)

        loss = torch.stack(losses).mean()
        mean_losses_epoch.append(loss.detach().numpy())

        print('Loss: ', loss)

        torch.save(model.state_dict(), model_path)

    plot_loss(mean_losses_epoch)
   

    test_dataset, test_targets = generate_dataset(100, 200, 1, 1)

#    test_model(model, test_dataset, test_targets)

    plot_vector_field(model, lagrangian_path(model, torch.tensor([0, 1], dtype=torch.float32, device=device, requires_grad=True)))
    
    return 0

if __name__ == '__main__':
    main()
    sys.exit()
