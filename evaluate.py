from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path(__file__).parent / "models" / "mnist_cnn.keras"
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test loss:     {loss:.4f}\n")

    preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_test, preds, digits=4))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:")
    print(cm)

    try:
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        fig.colorbar(im)
        fig.tight_layout()
        out = RESULTS_DIR / "confusion_matrix.png"
        fig.savefig(out, dpi=150)
        print(f"\nSaved confusion matrix to {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
