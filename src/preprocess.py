import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def normalize(x):
    return x.astype("float32") / 255.0 * 2.0 - 1.0


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train = np.load("data/raw/train.npz")
    test = np.load("data/raw/test.npz")

    x_train, x_val, y_train, y_val = train_test_split(
        normalize(train["x"]), train["y"],
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )
    x_test, y_test = normalize(test["x"]), test["y"]

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_train, y=y_train)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test, y=y_test)
    print(f"train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")


if __name__ == "__main__":
    main()
