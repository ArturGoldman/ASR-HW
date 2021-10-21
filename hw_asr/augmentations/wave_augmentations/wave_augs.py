import torch.distributions
import torch_audiomentations
from torch import Tensor
import librosa
from hw_asr.augmentations.base import AugmentationBase
import random


class GaussianNoise(AugmentationBase):
    def __init__(self, std=0.05):
        self.dist = torch.distributions.Normal(0, std)

    def __call__(self, data: Tensor):
        out = data + self.dist.sample(data.size())
        return out


class RandomPitchShift(AugmentationBase):
    def __init__(self, p=0.2, sr=16000, n_steps=3., normaliser=2):
        self.p = p
        self.sr = sr
        self.n_steps = n_steps
        self.normaliser = normaliser

    def __call__(self, data: Tensor):
        q = random.random()
        if q < self.p:
            ps = self.n_steps
            if self.n_steps == "random":
                ps = (random.random()-0.5)/self.normaliser
            data = librosa.effects.pitch_shift(data.numpy().squeeze(), self.sr, ps)
        return torch.Tensor(data).reshape(1, -1)


class TimeStretching(AugmentationBase):
    def __init__(self, p=0.2, rate=2.):
        self.p = p
        self.rate = rate

    def __call__(self, data: Tensor):
        q = random.random()
        if q < self.p:
            data = librosa.effects.time_stretch(data.numpy().squeeze(), self.rate)
        return torch.tensor(data).reshape(1, -1)


class Gain(AugmentationBase):
    def __init__(self, sr=16000, *args, **kwargs):
        self.sr = sr
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        return torch.Tensor(self._aug(data, sample_rate=self.sr))
