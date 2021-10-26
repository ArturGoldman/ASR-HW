#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rDrST064UU-zkyRsa8XQP6OuVq0p3fDW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rDrST064UU-zkyRsa8XQP6OuVq0p3fDW" -O ./ASR-HW/default_test/model.pth && rm -rf /tmp/cookies.txt

pip install -r ./ASR-HW/requirements.txt

git clone -q --recursive https://github.com/parlance/ctcdecode.git
pip install ./ctcdecode

wget -q -O ./ASR-HW/lm.arpa.gz http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
gunzip ./ASR-HW/lm.arpa.gz

python3 ./ASR-HW/test.py -c ./ASR-HW/default_test/config.json -r ./ASR-HW/default_test/model.pth
