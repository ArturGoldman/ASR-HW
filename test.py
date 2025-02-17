import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import hw_asr.model as module_model
import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser
import torch.nn.functional as F
from hw_asr.metric.utils import calc_wer, calc_cer

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = config.init_obj(config["loss"], module_loss).to(device)
    metric_fns = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    metrics = {
        "WER (Argmax)": 0,
        "CER (Argmax)": 0,
        "WER (Beam-Search + LM shallow fusion):": 0,
        "CER (Beam-Search + LM shallow fusion):": 0
    }

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            outputs = model(**batch)
            if type(outputs) is dict:
                batch.update(outputs)
            else:
                batch["logits"] = outputs

            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            for i in range(len(batch["text"])):
                item = {
                    "ground_truth": batch["text"][i],
                    "pred_text_argmax": text_encoder.ctc_decode(batch["argmax"][i][:int(batch["log_probs_length"][i])]),
                    "pred_text_beam_search": text_encoder.ctc_beam_search(
                        batch["probs"][i][:int(batch["log_probs_length"][i])])[:10],
                }
                results.append(item)
                cur_metrics = {
                    "WER (Argmax)": calc_wer(item["ground_truth"], item["pred_text_argmax"]),
                    "CER (Argmax)": calc_cer(item["ground_truth"], item["pred_text_argmax"]),
                    "WER (Beam-Search + LM shallow fusion):": calc_wer(item["ground_truth"],
                                                                       item["pred_text_beam_search"][0][0]),
                    "CER (Beam-Search + LM shallow fusion):": calc_cer(item["ground_truth"],
                                                                       item["pred_text_beam_search"][0][0])
                }
                for key, num in cur_metrics.items():
                    metrics[key] += num
    for key, num in metrics.items():
        print(key, num/len(results))

    with Path(out_file).open('w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default='output.json',
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader"
    )

    config = ConfigParser.from_args(args)

    args = args.parse_args()
    if "test" not in config["data"]:
        # this part brobably contains bugs
        # i suggest you put test dataset in test config
        test_data_folder = Path(args.test_data_folder)
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": test_data_folder / "audio",
                            "transcription_dir": test_data_folder / "transcriptions",
                        }
                    }
                ]
            }
        }
    main(config, args.output)
