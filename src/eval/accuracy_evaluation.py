import torch
from tqdm import tqdm


def accuracy_evaluate(model, DS, batch_size=128):
    X, Y = [x.type(torch.float32).split(batch_size, 0) for x in DS]
    model.eval()
    accuracy_sum = 0
    for X_MB, Y_MB in tqdm(list(zip(X, Y))):
        P_MB = model(X_MB)
        accuracy_mb = ((P_MB.argmax() - Y_MB.argmax()) == 0).mean()
        accuracy_sum += accuracy_mb
    print(f'accuracy: {accuracy_sum / len(X)}')
