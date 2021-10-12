# Don't forget to support cases when target_text == ''
import editdistance


def calc_wer(target_text, predicted_text) -> float:
    n = len(target_text.split())
    if target_text == "":
        n = 1
    ans = editdistance.eval(target_text.split(), predicted_text.split()) / n
    return ans


def calc_cer(target_text, predicted_text) -> float:
    n = len(target_text)
    if target_text == "":
        n = 1
    ans = editdistance.eval(target_text, predicted_text) / n

    return ans
