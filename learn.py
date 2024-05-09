import torch

import lagrangian as lag

import matplotlib.pyplot as plt

def l2_loss(q_dot_dot_predicted, q_dot_dot_actual):
    return (q_dot_dot_predicted - q_dot_dot_actual).pow(2)

def epoch(model, dataset, targets, device):
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

def load_model_or_make_new(model_path, device):
    try:
        model = lag.LagrangianNN().to(device)
        model.load_state_dict(torch.load(model_path))
    except:
        model = lag.LagrangianNN().to(device)
    return model

def train_loop(model, lr, n, dataset_generator, device, iter_cb=None) -> int:
    mean_losses = []


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=300)

    for i in range(n):
        optimizer.zero_grad()
        dataset, targets = dataset_generator(device)
        losses = epoch(model, dataset, targets, device)

        loss = torch.stack(losses).mean()
        mean_losses.append(loss.detach().cpu().numpy())

        optimizer.step()
        scheduler.step(loss)

        print('Epoch: ', i, 'Loss: ', loss.item(), 'LR: ', optimizer.param_groups[0]['lr'])

        if not iter_cb is None:
            iter_cb(model, i, mean_losses)

    return 0

