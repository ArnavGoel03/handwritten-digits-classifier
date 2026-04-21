from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "mnist_cnn.keras"

BATCH_SIZE = 128
EPOCHS = 10
SEED = 42


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test)


def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final val accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
