from math import pi
import torch

import numpy as np

from integrate_motion import lagrangian_path, plot_vector_field

import random

import sys

import matplotlib.pyplot as plt

import numpy as np

import torch

from learn import load_model_or_make_new, train_loop

def harmonic_osillator(t, amplitude, phi):
    omega = 1
    return amplitude * np.cos(omega * t + phi), -amplitude * omega * np.sin(omega * t + phi), -amplitude * omega**2 * np.cos(omega * t + phi)

def harmonic_oscillator_generate_path(t_range_start, t_range_end, dt, amplitude, phi, device):
    dataset = []
    targets = []

    for t in np.arange(t_range_start, t_range_end, dt):
        q, q_dot, q_dot_dot = harmonic_osillator(t, amplitude, phi)

        dataset.append([q, q_dot])
        targets.append(q_dot_dot)
        
    dataset = torch.tensor(dataset, dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    return dataset, targets

def generate_harmonic_oscillator_dataset(device):
    random_amplitude = random.uniform(-2, 2)
    random_phi = random.uniform(-2*pi, 2*pi)

    path_start = random.randint(0, 100)

    dataset, targets = harmonic_oscillator_generate_path(path_start, path_start + 30, 1, random_amplitude, random_phi, device)

    return dataset, targets

def main():
    device = 'cpu'
    model_path = 'model_softplus.pth'

    model = load_model_or_make_new(model_path, device)
    train_loop(model, 1e-3, 1000, generate_harmonic_oscillator_dataset, device)

    plt.cla()

    ground_truth_path, _ = harmonic_oscillator_generate_path(0, 2*pi, 0.1, 1, 0, device)

    predicted_path = lagrangian_path(model, torch.tensor([0, 1], dtype=torch.float32, device=device, requires_grad=True))

    plot_vector_field(model, predicted_path.detach(), ground_truth_path.detach())

if __name__ == '__main__':
    main()
    sys.exit()
