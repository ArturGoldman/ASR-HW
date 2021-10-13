from torch import nn
from torch.nn import Sequential, LSTM, Linear

from hw_asr.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return self.net(spectrogram)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


class BasicLSTM(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=3, hidden_size=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = LSTM(n_feats, hidden_size, n_layers, dropout=0.1, batch_first=True)
        self.tail = nn.Sequential(
            Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            Linear(hidden_size // 2, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        output, _ = self.net(spectrogram)
        output = nn.ReLU()(output)
        output = self.tail(output)
        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


class BasicGRU(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=3, hidden_size=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = nn.GRU(n_feats, hidden_size, n_layers, dropout=0.1, batch_first=True)
        self.tail = nn.Sequential(
            Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            Linear(hidden_size // 2, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        output, _ = self.net(spectrogram)
        output = nn.ReLU()(output)
        output = self.tail(output)
        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
