from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

from ctcdecode import CTCBeamDecoder

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 50) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        labels = ""
        for i in range(len(self.ind2char)):
            labels += self.ind2char[i]
        decoder = CTCBeamDecoder(
            labels,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=beam_size,
            num_processes=4,
            blank_id=0,
            log_probs_input=True
        )
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(probs)
        print(beam_results, beam_scores)
        hypos = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x[1], reverse=True)
