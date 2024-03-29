import numpy as np
import torch


def train(dataloader, model, loss_fn, optimizer, device='cuda'):
    model.train()
    loss, test_loss, acc = None, None, None
    for ft1, ft2, flag in dataloader:
        ft1, ft2, flag = ft1.to(device), ft2.to(device), flag.to(device)

        # Compute prediction error
        embs = model(ft1, ft2)
        loss = loss_fn(embs, flag.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()

    test_loss, acc = test(dataloader, model, loss_fn, device=device)

    return round(loss, 4), round(test_loss, 4), round(acc, 3)


def test(dataloader, model, loss_fn, device='cuda'):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for ft1, ft2, flag in dataloader:
            ft1, ft2, flag = ft1.to(device), ft2.to(device), flag.to(device)
            embs = model(ft1, ft2)
            test_loss += loss_fn(embs, flag.float()).item()

            if dataloader.dataset.onehot:
                predictions = torch.argmax(embs, dim=1)
                flag = torch.argmax(flag, dim=1)
            else:
                predictions = torch.round(embs)

            correct += torch.count_nonzero(flag == predictions).item() / predictions.shape[0]

    test_loss /= num_batches
    acc = correct / num_batches * 100
    return round(test_loss, 4), round(acc, 3)
