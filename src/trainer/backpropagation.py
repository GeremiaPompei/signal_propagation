import torch
from tqdm.notebook import tqdm


def train_backpropagation(model, TR_SET, epochs=10, batch_size=1024):
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    TR_X, TR_Y = [x.type(torch.float32).split(batch_size, 0) for x in TR_SET]
    model.train()
    for epoch in tqdm(range(epochs)):
        tr_loss_sum = 0
        for TR_X_MB, TR_Y_MB in zip(TR_X, TR_Y):
            optim.zero_grad()
            TR_P_MB = model(TR_X_MB)
            loss = criterion(TR_P_MB, TR_Y_MB)
            tr_loss_sum += loss.item()
            loss.backward()
            optim.step()
        tqdm.write(f'epoch: {epoch + 1}/{epochs} - tr_loss: {tr_loss_sum / len(TR_X)}')
