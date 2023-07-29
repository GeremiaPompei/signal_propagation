import torch
from tqdm.notebook import tqdm


def crossentropy_evaluate(label, model, DS, batch_size=128):
    model.eval()
    X, Y = [x.type(torch.float32).split(batch_size, 0) for x in DS]
    ce_sum = 0
    for X_MB, Y_MB in tqdm(list(zip(X, Y))):
        P_MB = model(X_MB)
        ce_sum -= (Y_MB * torch.nn.functional.softmax(P_MB, dim=1).log()).mean(0).sum()
    print(f'{label} cross-entropy: {ce_sum / len(X)}')
