import torch

import numpy as np

import lagrangian as lag

import sys

device = 'cpu'

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    losses = []
    
    for epoch in range(100):
        loss_total = torch.tensor(0.0, device=device, requires_grad=True)

        for i in range(len(dataset)):
            optimizer.zero_grad()
            q_dot_dot_predicted = lag.q_dot_dot(model, dataset[i])
            loss_i = l2_loss(q_dot_dot_predicted, targets[i])
            loss_total = loss_total +  loss_i

        loss_total.backward()
        optimizer.step()
        losses.append(loss_total.detach().numpy())
        print('Epoch: ', epoch, 'Loss: ', losses[-1].item())

def load_model_or_make_new():
    try:
        model = lag.LagrangianNN().to(device)
        model.load_state_dict(torch.load('model.pth'))
    except:
        model = lag.LagrangianNN().to(device)
    return model

def main() -> int:
    model = load_model_or_make_new()
    
    dataset, targets = generate_dataset(0, 1000)
    
#    train(model, dataset, targets)

#    torch.save(model.state_dict(), 'model.pth')
    
#    plot_loss(losses)

    test_dataset, test_targets = generate_dataset(100, 200)

    test_model(model, test_dataset, test_targets)

    plot_test(model, dataset, targets)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
