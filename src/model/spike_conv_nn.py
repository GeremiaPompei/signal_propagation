import torch.nn


class SpikeLayer:

    def __init__(self, times, spike_threshold, reset_value=0):
        self.times = times
        self.spike_threshold = spike_threshold
        self.reset_value = reset_value
        self.counter = 0
        self.V = None

    def __call__(self, H):
        if self.counter % self.times == 0:
            self.V = torch.zeros_like(H)
        self.V = self.V.detach() + H.detach()
        self.V[self.V.abs() > self.spike_threshold] = self.reset_value
        self.counter += 1
        return self.V


class ConvBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, times, spike_threshold):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1,  bias=False)
        self.bn = torch.nn.BatchNorm2d(planes)
        self.spike = SpikeLayer(times, spike_threshold)
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.spike(x)
        x = self.pooling(x)
        return x


class ConvSpikeNN(torch.nn.Module):

    def __init__(self, num_classes, in_channels=1, fc_size=128, times=4, spike_threshold=1):
        super().__init__()
        self.times = times
        self.conv1 = ConvBlock(in_channels, 20, times=times, spike_threshold=spike_threshold)
        self.conv2 = ConvBlock(20, 40, times=times, spike_threshold=spike_threshold)
        self.fc = torch.nn.Linear(1960, fc_size)
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
