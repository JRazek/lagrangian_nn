import torch

import numpy as np

from integrate_motion import lagrangian_path, plot_vector_field

import random

import lagrangian as lag

import sys

import matplotlib.pyplot as plt

device = 'cpu'
model_path = 'model_softplus.pth'

def harmonic_osillator(t, amplitude, phi):
    omega = 1
    return amplitude * np.cos(omega * t + phi), -amplitude * omega * np.sin(omega * t + phi), -amplitude * omega**2 * np.cos(omega * t + phi)

def l2_loss(q_dot_dot_predicted, q_dot_dot_actual):
    return (q_dot_dot_predicted - q_dot_dot_actual).pow(2)

def generate_dataset(range_start, range_end, random_amplitude, random_phi):
    dataset = []
    targets = []
    for t in range(range_start, range_end):
        q, q_dot, q_dot_dot = harmonic_osillator(t, random_amplitude, random_phi)

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
    plt.savefig('loss.png')

def epoch(model, dataset, targets):
    losses = []
    
    loss_total = torch.tensor(0.0, device=device, requires_grad=True)

    for i in range(len(dataset)):
        q_dot_dot_predicted = lag.q_dot_dot(model, dataset[i])

        assert not q_dot_dot_predicted.isnan().any()

        loss_i = l2_loss(q_dot_dot_predicted, targets[i])
        loss_total = loss_total + loss_i

    loss_total.backward()
        
    losses.append(loss_total)

    return losses

def load_model_or_make_new():
    try:
        model = lag.LagrangianNN().to(device)
        model.load_state_dict(torch.load(model_path))
    except:
        model = lag.LagrangianNN().to(device)
    return model

def train_loop(model, lr, n) -> int:
    
    mean_losses_epoch = []


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=800)

    for i in range(n):
        optimizer.zero_grad()
        random_amplitude = random.uniform(-2, 2)
        random_phi = random.uniform(-1, 1)

        random_n = random.randint(0, 100)

        dataset, targets = generate_dataset(random_n, random_n + 30, random_amplitude, random_phi)
        losses = epoch(model, dataset, targets)

        loss = torch.stack(losses).mean()
        mean_losses_epoch.append(loss.detach().cpu().numpy())

        optimizer.step()
        scheduler.step(loss)

        print('Epoch: ', i, 'Loss: ', loss.item(), 'LR: ', optimizer.param_groups[0]['lr'])

        if (i+1) % 200 == 0:
            torch.save(model.state_dict(), model_path)

        if (i+1) % 1000 == 0:
            plt.cla()
            plot_loss(mean_losses_epoch)
    
    return 0

def main():
    model = load_model_or_make_new()
    train_loop(model, 1e-3, 100)

    plt.cla()
    plot_vector_field(model, lagrangian_path(model, torch.tensor([0, 1], dtype=torch.float32, device=device, requires_grad=True)))

if __name__ == '__main__':
    main()
    sys.exit()
