# Fashion-MNIST ANN Pipeline
Fashion-MNIST ANN: end-to-end ML versioning with Git, DVC and Google Drive (fully-connected ANN, reproducible with dvc repro).

## Structure
- src/prepare.py: download and save raw data
- src/preprocess.py: normalize and split
- src/train.py: build and train the ANN
- src/evaluate.py: test metrics and confusion matrix
- params.yaml: hyperparameters
