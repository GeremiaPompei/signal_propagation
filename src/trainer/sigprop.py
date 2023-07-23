import torch
from tqdm.notebook import tqdm


def get_leaf_layers(m, device='cpu'):
    children = list(m.children())
    if not children:
        return [m.to(device)]
    leaves = []
    for l in children:
        leaves.extend(get_leaf_layers(l, device=device))
    return leaves


def train_sigprop(model, TR_SET, epochs=10, batch_size=128, device='cpu'):
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    TR_X, TR_Y = [x.type(torch.float32).split(batch_size, 0) for x in TR_SET]
    layers = get_leaf_layers(model, device=device)
    dim_o = layers[0].out_channels
    dim_w, dim_h = TR_X[0].shape[2], TR_X[0].shape[3]
    output_embedding_layer = torch.nn.Linear(TR_Y[0].shape[1], dim_o * dim_w * dim_h).to(device)
    model.train()
    for epoch in range(epochs):
        tr_loss_sum = 0
        for TR_X_MB, TR_Y_MB in tqdm(list(zip(TR_X, TR_Y))):
            h, t = TR_X_MB, TR_Y_MB
            layers_loss = []
            for i, layer in enumerate(layers):
                if i > 0:
                    cat_ht = torch.cat((h, t)).squeeze()
                    h_n, t_n = layer(cat_ht).split(h.shape[0])
                else:
                    h_n = layer(h)
                    t_n = output_embedding_layer(t).view(-1, dim_o, dim_w, dim_h)
                optim.zero_grad()
                loss = (t_n - h_n).abs().mean()
                try:
                    loss.backward()
                    optim.step()
                    h, t = h_n.detach(), t_n.detach()
                    layers_loss.append(loss.item())
                except:
                    h, t = h_n, t_n
            tr_loss_sum += torch.Tensor(layers_loss).mean()
        print(f'epoch: {epoch + 1}/{epochs} - tr_loss: {tr_loss_sum / len(TR_X)}')
