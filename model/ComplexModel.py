import torch
from torchcomplex import nn
from torch.autograd import Function


class Exponential(Function):
    @staticmethod
    def forward(ctx, x):
        expx = torch.exp(x)
        ctx.save_for_backward(expx)
        return expx

    @staticmethod
    def backward(ctx, grad_output):
        expx, = ctx.saved_tensors
        return expx * grad_output

    @staticmethod
    def exp(x):
        return Exponential.apply(x)


class PaddedLayer(nn.Module):
    def __init__(self, in_var=1, features=40, complex_weights=False):
        super().__init__()
        self.layer = nn.Linear(in_var, features, complex_weights=complex_weights)

    def forward(self, x):
        forwarded = self.layer(x)
        expForward = Exponential.exp(forwarded)
        assert forwarded.shape == expForward.shape
        return expForward


class LayeredModel(nn.Module):
    def __init__(self, in_var=1, final_var=4, features=40, layer_count=1, p=0.35, act=nn.CReLU):
        super().__init__()
        self.p = p
        self.act = act
        layers = [PaddedLayer(in_var, features, complex_weights=True), ]
        for i in range(layer_count):
            layers.extend(self._layer(features))
        self.layers = nn.Sequential(*layers)
        self.layers.append(PaddedLayer(features, final_var))

    def forward(self, x: torch.Tensor):
        return self.layers(x.type(torch.complex64)).real

    def _layer(self, features):
        return [PaddedLayer(features, features, complex_weights=True), self.act(), nn.Dropout(self.p)]


class SingleApprox(nn.Module):
    def __init__(self, in_var=1, features=40, final_var=1, complex_weights=True):
        super().__init__()
        layer = PaddedLayer(in_var=in_var, features=features, complex_weights=complex_weights)
        final_layer = nn.Linear(features, final_var, complex_weights=complex_weights)
        self.layers = nn.Sequential(layer, final_layer)

    def forward(self, x: torch.Tensor):
        return self.layers(x.type(torch.complex64)).real
