# ASR-HW

Implementing [QuartzNet](https://arxiv.org/abs/1910.10261)

## Test running guide

To run test 
```
!git clone https://github.com/ArturGoldman/ASR-HW
!chmod u+x ./ASR-HW/test.sh
!./ASR-HW/test.sh
```

If ran on Yandex DataSphere, last comand should be run in different cell using GPU, e.g.
```
#!g1.1:bash
./ASR-HW/test.sh
```

After running test, output.json file should be created, and metric values should be printed.

Warning: test.sh should be run on linux. Comand execution went well on Yandex DataSphere and Google Colab, but had bad time on Mac OS.

## What should be noted
By default configs assume, that training and testing is run on DataSphere with bucket bucket-hse-rw with LibriSpeech data. If directory is different, "data_dir" should be changed in configs

## Results
Managed to achieve following metric values on test-clean by training QuartzNet5x5 on train-clean-100, using Adam.

- WER (Argmax): 0.7479754829078634
- CER (Argmax): 0.34923065666638836
- WER (Beam-Search + LM shallow fusion): 0.6314003097260577
- CER (Beam-Search + LM shallow fusion): 0.3431540827746051

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
