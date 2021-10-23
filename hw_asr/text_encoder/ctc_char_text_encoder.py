from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

from ctcdecode import CTCBeamDecoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    lm_model = './ASR-HW/lm.arpa'
    beam_size = 100

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        labels = ""
        for i in range(len(self.ind2char)):
            labels += self.ind2char[i].upper()
        self.decoder = CTCBeamDecoder(
            labels,
            model_path=self.lm_model,
            alpha=0.5,
            beta=1,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=self.beam_size,
            num_processes=4,
            blank_id=0,
            log_probs_input=False
        )

    def ctc_decode(self, inds: List[int]) -> str:
        if len(inds) == 0:
            return ""

        if torch.is_tensor(inds):
            inds = inds.tolist()
        ans_str = self.ind2char[inds[0]]
        i = 0
        while i != len(inds):
            if ans_str[-1] != self.ind2char[inds[i]]:
                ans_str += self.ind2char[inds[i]]
            i += 1
        ans_str = ans_str.replace(self.EMPTY_TOK, '')
        return ans_str

    def ctc_beam_search(self, probs: torch.tensor) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).

        probs: [N_TIMESTEPS x N_VOCAB] - probabilities of tokens. they should sum up to 1
        """

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(probs.unsqueeze(0))
        hypos = []
        for k in range(self.beam_size):
            hypos.append((self.ctc_decode(beam_results[0][k][:out_lens[0][k]]), beam_scores[0][k].item()))
        return sorted(hypos, key=lambda x: x[1], reverse=True)
