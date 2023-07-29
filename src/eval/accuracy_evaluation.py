import torch
from tqdm.notebook import tqdm


def accuracy_evaluate(label, model, DS, batch_size=128):
    model.eval()
    X, Y = [x.type(torch.float32).split(batch_size, 0) for x in DS]
    accuracy_sum = 0
    for X_MB, Y_MB in tqdm(list(zip(X, Y))):
        P_MB = model(X_MB)
        accuracy_mb = ((P_MB.argmax(1) - Y_MB.argmax(1)) == 0).type(torch.float).mean()
        accuracy_sum += accuracy_mb
    print(f'{label} accuracy: {accuracy_sum / len(X)}')
