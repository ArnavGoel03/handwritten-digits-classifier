import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "models" / "mnist_cnn.keras"


def load_and_preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img) if np.asarray(img).mean() > 127 else img
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr[np.newaxis, ..., np.newaxis]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py <path-to-image>")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    x = load_and_preprocess(image_path)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    print(f"Predicted digit: {digit}  (confidence: {probs[digit]:.2%})")
    ranked = sorted(enumerate(probs), key=lambda p: -p[1])
    print("Top 3:")
    for d, p in ranked[:3]:
        print(f"  {d}: {p:.2%}")


if __name__ == "__main__":
    main()
