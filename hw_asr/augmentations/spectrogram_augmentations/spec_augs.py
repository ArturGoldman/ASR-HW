from hw_asr.augmentations.base import AugmentationBase
from torch import Tensor
import torch
from torchvision.transforms import Normalize
import random
import torch.nn as nn
import torchaudio.transforms


class NormalizeAug(AugmentationBase):
    def __init__(self, normalize):
        if normalize == 'to05':
            self.normalize = Normalize([0.5], [0.5])
        elif normalize == 'touniform':
            self.normalize = lambda x: (x - torch.mean(x, dim=1, keepdim=True)) / (
                        torch.std(x, dim=1, keepdim=True) + 1e-18)
        else:
            self.normalize = None

    def __call__(self, melspec: Tensor):
        out = torch.log(torch.clamp(melspec, min=1e-18))
        if self.normalize is not None:
            out = self.normalize(out)
        return out


class MaskAug(AugmentationBase):
    def __init__(self, p=0.5, time_percent=0.15, freq_percent=0.15):
        self.p = p
        self.tp = time_percent
        self.fp = freq_percent

    def __call__(self, melspec: Tensor):
        q = random.random()
        if q < self.p:
            f, t = melspec.size()[-2:]
            specaug = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(int(f*self.fp)),
                torchaudio.transforms.TimeMasking(int(t*self.tp)),
            )
            melspec = specaug(melspec)
        return melspec
