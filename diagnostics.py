import math
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Import your existing project files
from model import get_unet_resnet50
from utility import set_seed

EXPERIMENT_LOG = []
NUM_CLASSES = 6


# ==========================================
# PART 1: THE FOUR SANITY CHECKS
# ==========================================

def check_data(train_loader, val_loader):
    """Check A: per-channel statistics of both splits."""
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        x, _ = next(iter(loader))
        m = x.mean(dim=(0, 2, 3)).tolist()
        s = x.std(dim=(0, 2, 3)).tolist()
        print(f'{split:>5}: mean per channel = {[round(v, 3) for v in m[:3]]}... (showing first 3 of 13)')
        print(f'       std per channel = {[round(v, 3) for v in s[:3]]}... ')
        print(f'       min = {x.min().item():.2f}, max = {x.max().item():.2f}')


def check_split(train_dataset, val_dataset):
    """Check B: are any files in both splits?"""
    # Assumes dataset.images contains the file paths
    overlap = set(train_dataset.images) & set(val_dataset.images)
    print(f'Train size {len(train_dataset)}, val size {len(val_dataset)}, overlap: {len(overlap)} examples')
    if overlap:
        print(f'WARNING: Overlap detected!')


@torch.no_grad()
def check_initial_loss(model, train_loader, loss_fn, device):
    """Check C: loss of the untrained model vs. the expected ln(C)."""
    model.eval()
    x, y, _ = next(iter(train_loader))
    loss = loss_fn(model(x.to(device)), y.to(device)).item()
    print(f'Initial loss: {loss:.3f}   (expected ≈ ln({NUM_CLASSES}) = {math.log(NUM_CLASSES):.3f})')


def overfit_one_batch(model, train_loader, loss_fn, device, steps=100, lr=0.01):
    """Check D: can the model memorise a single small batch?"""
    xb, yb, _ = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    losses = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Step');
    plt.ylabel('Training loss (log scale)');
    plt.title('Loss on one batch')
    plt.grid(alpha=0.3)
    plt.show()
    print(f'Final loss: {losses[-1]:.4f}')


# ==========================================
# PART 2: DIAGNOSTICS (ACTIVATIONS & GRADIENTS)
# ==========================================

@torch.no_grad()
def activation_stats(model, x, layer_type=nn.ReLU):
    """Records the output of every `layer_type` layer for the batch x."""
    model = copy.deepcopy(model).train()
    records, hooks = [], []
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            hooks.append(module.register_forward_hook(
                lambda m, inp, out, name=name: records.append((name, out.detach().float().cpu()))))
    model(x)
    for h in hooks:
        h.remove()
    return records


def plot_activation_histograms(model, x, title=''):
    """One histogram per ReLU layer, plus mean/std and the fraction of dead units."""
    records = activation_stats(model, x)
    # Limit to first 6 layers to avoid plotting 50+ ResNet ReLUs
    records = records[:6]
    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 3), sharey=True)
    for i, (ax, (name, act)) in enumerate(zip(axes, records)):
        vals = act.flatten().numpy()
        per_channel_max = act.transpose(0, 1).reshape(act.shape[1], -1).max(dim=1).values
        dead = (per_channel_max <= 0).float().mean().item()
        ax.hist(vals, bins=40, color='steelblue')
        ax.set_yscale('log')
        ax.set_title(f'ReLU {i + 1}\nmean {vals.mean():.2g}\nstd {vals.std():.2g}\ndead {dead:.0%}', fontsize=8)
    axes[0].set_ylabel('count (log)')
    plt.suptitle(f'Activation histograms (First 6 ReLUs)  {title}', y=1.1)
    plt.show()


def plot_grad_flow(model, x, y, loss_fn, title=''):
    """Mean absolute gradient of the loss w.r.t. the weights of each conv/linear layer."""
    model.train()
    model.zero_grad()
    loss_fn(model(x), y).backward()
    names, grads = [], []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
            grads.append(module.weight.grad.abs().mean().item())
    model.zero_grad()

    # Plot only the first 20 layers for readability
    grads = grads[:20]
    plt.figure(figsize=(7, 3))
    plt.bar(range(len(grads)), grads, color='coral')
    plt.yscale('log')
    plt.xlabel('Layer (input → output)')
    plt.ylabel('mean |∂loss/∂W| (log scale)')
    plt.title(f'Gradient flow (First 20 Layers) {title}')
    plt.grid(alpha=0.3, axis='y')
    plt.show()


# ==========================================
# PART 3: THE EXPERIMENT LOGGER
# ==========================================

@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss, seen = 0.0, 0
    for xb, yb, _ in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(xb)
        seen += len(xb)
    return total_loss / seen


def run_experiment(name, train_loader, val_loader, device, lr=1e-4, epochs=3, seed=0, log=True, verbose=True):
    """Trains one model from scratch and logs the result to EXPERIMENT_LOG."""
    config = dict(name=name, lr=lr, epochs=epochs, seed=seed)
    set_seed(seed)

    model = get_unet_resnet50(in_channels=13, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {'iter_loss': [], 'train_loss': [], 'val_loss': []}
    diverged = False
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0

        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

            if not torch.isfinite(loss):
                diverged = True
                break

            history['iter_loss'].append(loss.item())
            running += loss.item() * len(xb)
            seen += len(xb)

        if diverged:
            if verbose: print(f'  [{name} | seed {seed}] diverged in epoch {epoch + 1}')
            break

        val_loss = evaluate(model, val_loader, loss_fn, device)
        history['train_loss'].append(running / seen)
        history['val_loss'].append(val_loss)

    runtime = time.time() - start

    result = dict(config,
                  diverged=diverged,
                  train_loss=history['train_loss'][-1] if history['train_loss'] else float('nan'),
                  val_loss=history['val_loss'][-1] if history['val_loss'] else float('nan'),
                  runtime_s=round(runtime, 1))

    if log:
        EXPERIMENT_LOG.append(result)
        pd.DataFrame(EXPERIMENT_LOG).to_csv('experiment_log.csv', index=False)

    if verbose and not diverged:
        print(f'  [{name} | seed {seed}] train loss {result["train_loss"]:.3f} | '
              f'val loss {result["val_loss"]:.3f} | {runtime:.0f}s')

    return model, history


def summarise(names):
    """Mean ± std over seeds for the named experiments, from the log."""
    df = pd.DataFrame(EXPERIMENT_LOG)
    df = df[df['name'].isin(names)]
    out = df.groupby('name', sort=False).agg(
        runs=('seed', 'count'),
        val_loss_mean=('val_loss', 'mean'),
        val_loss_std=('val_loss', 'std'),
        train_loss_mean=('train_loss', 'mean'),
        runtime_s=('runtime_s', 'mean')
    )
    return out.reindex(names).round(4)