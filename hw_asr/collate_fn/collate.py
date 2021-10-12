import logging
from typing import List

import torch
from hw_asr.base.base_text_encoder import BaseTextEncoder

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    if len(dataset_items) == 0:
        raise Exception('Received empty list')

    spectrogram_lengths = []
    text_enc_lengths = []
    initial_normalised_encodings = []
    for i in range(len(dataset_items)):
        item = dataset_items[i]
        spectrogram_lengths.append(item['spectrogram'].size()[-1])
        text_enc_lengths.append(item['text_encoded'].size()[-1])
        initial_normalised_encodings.append(BaseTextEncoder.normalize_text(item['text']))

    max_sp_len = max(spectrogram_lengths)
    max_enc_len = max(text_enc_lengths)

    target_shape = dataset_items[0]['spectrogram'].size()

    spec_out = torch.zeros((len(dataset_items), target_shape[1], max_sp_len))
    text_enc_out = torch.zeros((len(dataset_items), max_enc_len))

    for i in range(len(dataset_items)):
        item = dataset_items[i]
        spec_out[i, :, : spectrogram_lengths[i]] = item['spectrogram']
        text_enc_out[i, :text_enc_lengths[i]] = item['text_encoded']

    return {
        'spectrogram':spec_out,
        'text_encoded':text_enc_out.int(),
        'text_encoded_length':torch.tensor(text_enc_lengths),
        'text':initial_normalised_encodings
    }
