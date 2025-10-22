import torch.nn as nn

from .layers import ResidualBlock, RefineBlock
from .normalization import InstanceNorm2dPlus


class RefineNet(nn.Module):
    """
    Noise Conditional Score Network - Version 2
    
    Related Articles:
    - NCSN: Generative Modeling by Estimating Gradients of the Data Distribution, https://arxiv.org/abs/1907.05600;
    - NCSNv2: Improved Techniques for Training Score-Based Generative Models, https://arxiv.org/abs/2006.09011;
    - RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation, https://arxiv.org/abs/1611.06612.
    """
    def __init__(self, ngf, num_classes, sigmas, act=nn.ELU(), norm=InstanceNorm2dPlus, data_channels=3):
        super().__init__()
        self.norm = norm
        self.ngf = ngf
        self.num_classes = num_classes

        self.act = act
        self.register_buffer('sigmas', sigmas)

        self.begin_conv = nn.Conv2d(data_channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, data_channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, adjust_padding=False, dilation=4),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        h = 2 * x - 1 # !!IMPORTANT: Normalize pixel values to [-1, 1]

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output
