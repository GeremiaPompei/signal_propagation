import torch.nn


class SpikeVoltageLayer:

    def __init__(self, times, spike_threshold, reset_value=0):
        self.times = times
        self.spike_threshold = spike_threshold
        self.reset_value = reset_value
        self.counter = 0
        self.V = None

    def __call__(self, H):
        if self.counter % self.times == 0:
            self.V = torch.zeros_like(H)
        self.V = self.V + H.detach()
        self.V[self.V.abs() > self.spike_threshold] = self.reset_value
        self.counter += 1
        return self.V.detach()


class SurrogateLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, H):
        return 1 / torch.pi * torch.arctan(torch.pi * H) + 1 / 2


class ConvBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, voltage, times, spike_threshold):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(planes)
        self.spike = SpikeVoltageLayer(times, spike_threshold) if voltage else SurrogateLayer()
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.spike(x)
        x = self.pooling(x)
        return x


class ConvSpikeNN(torch.nn.Module):

    def __init__(
            self,
            num_classes,
            in_channels=1,
            conv_latent1=20,
            conv_latent2=40,
            fc_size=128,
            voltage=False,
            times=4,
            spike_threshold=1
    ):
        super().__init__()
        self.times = times
        self.conv1 = ConvBlock(
            in_channels,
            conv_latent1,
            voltage=voltage,
            times=times,
            spike_threshold=spike_threshold
        )
        self.conv2 = ConvBlock(
            conv_latent1,
            conv_latent2,
            voltage=voltage,
            times=times,
            spike_threshold=spike_threshold
        )
        self.fc = torch.nn.Linear(conv_latent2 * 49, fc_size)
        self.classifier = torch.nn.Linear(fc_size, num_classes)

    def forward(self, x):
        acc, counter = 0, 0
        for t in range(self.times):
            out = self.conv1(x)
            out = self.conv2(out)
            out = out.view(out.shape[0], -1)
            out = torch.relu(self.fc(out))
            acc += self.classifier(out)
            counter += 1
        return acc / counter
