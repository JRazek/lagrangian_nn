import torch

from itertools import product

from integrate_motion import differential

def plot_vector_field(model, path_prediction, path_ground_truth, device, path):
    import matplotlib.pyplot as plt

    q = torch.linspace(-2, 2, 10)
    q_dot = torch.linspace(-2, 2, 10)

    fig = plt.figure(figsize=(10, 10))

    for q_val, q_dot_val in product(q, q_dot):
        x = torch.tensor([q_val, q_dot_val], dtype=torch.float32, requires_grad=True, device=device)
        dx = differential(model, x)

        plt.quiver(q_val, q_dot_val, dx[0], dx[1], scale=30)

    plt.xlabel('q')
    plt.ylabel('q_dot')
    plt.tight_layout()
    
    plt.plot(path_prediction[:, 0], path_prediction[:, 1], color='red')
    plt.plot(path_prediction[0, 0], path_prediction[0, 1], 'ro')
    plt.text(path_prediction[0, 0], path_prediction[0, 1], 'Predicted', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='left', rotation=0)

    plt.plot(path_ground_truth[:, 0], path_ground_truth[:, 1], color='blue', linestyle='--')
    plt.plot(path_ground_truth[0, 0], path_ground_truth[0, 1], 'bo')
    plt.text(path_ground_truth[0, 0], path_ground_truth[0, 1], 'Ground truth', fontsize=12, color='blue', verticalalignment='bottom', horizontalalignment='right', rotation=0)

    fig.savefig(path)
    
def plot_loss(losses, path):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig(path)

