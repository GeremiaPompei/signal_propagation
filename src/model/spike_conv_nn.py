import torch


class SpikeActivation(torch.nn.Module):

    def __init__(self, times=4, spike_threshold=1, reset_value=0):
        super().__init__()
        self.times = times
        self.spike_threshold = spike_threshold
        self.reset_value = reset_value

    def forward(self, inputs):
        inputs = inputs.split(inputs.shape[0] // self.times)
        outputs = []
        V = torch.zeros_like(inputs[0])
        for I in inputs:
            V = V + I
            V[V.abs() > self.spike_threshold] = self.reset_value
            outputs.append(V)
        return torch.vstack(outputs)


class SurrogateActivation(torch.nn.Module):

    def __init__(self, spike_threshold=1, reset_value=0):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.reset_value = reset_value

    def forward(self, inputs):
        x = 1 / torch.pi * torch.arctan(torch.pi * inputs) + 1 / 2
        x[x.abs() > self.spike_threshold] = self.reset_value
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, activation_constructor):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(planes)
        self.activation = activation_constructor()
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
            activation_constructor=SurrogateActivation,
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            conv_latent1,
            activation_constructor=activation_constructor,
        )
        self.conv2 = ConvBlock(
            conv_latent1,
            conv_latent2,
            activation_constructor=activation_constructor,
        )
        self.fc = torch.nn.Linear(conv_latent2 * 49, fc_size)
        self.times = self.conv1.activation.times if type(self.conv1.activation) == SpikeActivation else 1
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
