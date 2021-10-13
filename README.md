# ASR project barebones

## overfit_version branch

This branch contains version of homework, which can overfit on batch of size 20. All files, connected to overfitting can be found in folder **overfit_log**:
- DLA_HW1_overfit.ipynb: this jupyter notebook can be run and it performs the whole overfitting process
- config.json: config, used to launch overfitting. The same config can be found as ASR-HW/hw-asr/my_overfit_config.json
- events.out.tfevents: tensorboard file, which contains all training info
- info.log: file that has all console output

---
The model used for overfitting was two-layered GRU with feed-forward network.
