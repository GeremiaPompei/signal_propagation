import torch


class SpikingActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, model):
        ctx.save_for_backward(inputs)
        inputs = inputs.split(inputs.shape[0] // model.times)
        outputs = []
        V = torch.zeros_like(inputs[0])
        for I in inputs:
            V = V + I
            V[V.abs() > model.spike_threshold] = model.reset_value
            outputs.append(V)
        return torch.vstack(outputs)

    @staticmethod
    def backward(ctx, grad_output):
        surrogate_res = 1 / (1 + (torch.pi * grad_output).pow(2))
        return surrogate_res, None


class SpikeLayer(torch.nn.Module):

    def __init__(self, times, spike_threshold, reset_value=0):
        super().__init__()
        self.times = times
        self.spike_threshold = spike_threshold
        self.reset_value = reset_value

    def __call__(self, inputs):
        return SpikingActivation.apply(inputs, self)


class ConvBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, times, spike_threshold):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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

    def __init__(
            self,
            num_classes,
            in_channels=1,
            conv_latent1=20,
            conv_latent2=40,
            fc_size=128,
            times=4,
            spike_threshold=1
    ):
        super().__init__()
        self.times = times
        self.conv1 = ConvBlock(
            in_channels,
            conv_latent1,
            times=times,
            spike_threshold=spike_threshold
        )
        self.conv2 = ConvBlock(
            conv_latent1,
            conv_latent2,
            times=times,
            spike_threshold=spike_threshold
        )
        self.fc = torch.nn.Linear(conv_latent2 * 49, fc_size)
        self.classifier = torch.nn.Linear(fc_size, num_classes)

    def forward(self, x):
        out = x.repeat(self.times, 1, 1, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        out = torch.relu(self.fc(out))
        out = self.classifier(out)
        out = torch.stack(out.split(out.shape[0] // self.times)).mean(0)
        return out
