import torch


class SpikingActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return (inputs > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return 1 / (1 + (torch.pi * grad_output).pow(2))


class SpikeLayer(torch.nn.Module):

    def __init__(self, times=4, spike_threshold=1, surrogate=True):
        super().__init__()
        self.times = times
        self.spike_threshold = spike_threshold
        self.surrogate = surrogate
        if self.surrogate:
            self.fire = SpikingActivation.apply
        else:
            self.fire = lambda x: SpikingActivation.forward(None, x)

    def forward(self, inputs):
        bs = inputs.shape[0] // self.times
        outputs = []
        V = 0
        for t in range(self.times):
            start, end = t * bs, (t + 1) * bs
            I = inputs[start: end]
            V = V + I - self.spike_threshold
            outputs.append(self.fire(V))
        return torch.vstack(outputs)


class ConvBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, times, spike_threshold, surrogate):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(planes)
        self.activation = SpikeLayer(
            times=times,
            spike_threshold=spike_threshold,
            surrogate=surrogate
        )
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x


class Classifier(torch.nn.Module):
    def __init__(self, fc_size, times, num_classes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand((num_classes, fc_size)))
        self.bias = torch.nn.Parameter(torch.rand(num_classes))
        self.times = times

    def forward(self, x):
        out = x @ self.W.T + self.bias
        return torch.stack(out.split(out.shape[0] // self.times)).mean(0)


class ConvSpikeNN(torch.nn.Module):

    def __init__(
            self,
            num_classes,
            in_channels=1,
            conv_latent1=20,
            conv_latent2=40,
            fc_size=128,
            times=4,
            spike_threshold=1,
            surrogate=True,
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            conv_latent1,
            times,
            spike_threshold,
            surrogate,
        )
        self.conv2 = ConvBlock(
            conv_latent1,
            conv_latent2,
            times,
            spike_threshold,
            surrogate,
        )
        self.fc = torch.nn.Linear(conv_latent2 * 49, fc_size)
        self.times = self.conv1.activation.times if type(self.conv1.activation) == SpikeLayer else 1
        self.classifier = Classifier(fc_size, self.times, num_classes)

    def preprocess(self, x):
        return x.repeat(self.times, *[1 for _ in range(len(x.shape) - 1)])

    def forward(self, x):
        out = self.preprocess(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out
