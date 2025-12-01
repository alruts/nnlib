import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 3 = errors only

import datetime
import hashlib
import io
import uuid

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorboard.plugins.hparams import api as hp


class TensorboardLogger:
    """
    Flexible TensorBoard logger compatible with JAX.
    Supports scalars, histograms, text, embeddings, custom plots, and HParams.
    Each experiment folder includes a unique hash to prevent overwriting runs.
    """

    def __init__(self, log_dir="runs", experiment_name=None, hash_len=8):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_name = experiment_name or "exp"

        raw = f"{experiment_name}{timestamp}{uuid.uuid4().hex}"
        exp_hash = hashlib.sha1(raw.encode()).hexdigest()[:hash_len]

        self.exp_name = f"{experiment_name}_{exp_hash}"
        self.log_dir = os.path.join(log_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = tf.summary.create_file_writer(self.log_dir)
        print(f"  TensorBoard logging initialized at: {self.log_dir}")

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def log_scalars(self, main_tag, tag_value_dict, step):
        with self.writer.as_default():
            for tag, val in tag_value_dict.items():
                tf.summary.scalar(f"{main_tag}/{tag}", val, step=step)
            self.writer.flush()

    def log_histogram(self, tag, values, step, buckets=100):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=buckets)
            self.writer.flush()

    def log_plot(self, tag, plot_fn, data, step):
        fig = plot_fn(data)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = np.array(Image.open(buf))[None, ...]  # (1, H, W, C)
        plt.close(fig)

        with self.writer.as_default():
            tf.summary.image(tag, image, step=step)
            self.writer.flush()

    def log_text(self, tag, text, step):
        with self.writer.as_default():
            tf.summary.text(tag, text, step=step)
            self.writer.flush()

    def log_hparams(self, hparams: dict, trial_id=None, start_time_secs=None):
        """
        Logs hyperparameters and optionally metrics for filtering in TensorBoard.

        Args:
            hparams (dict): Hyperparameters (name > value).
            trial_id (str): Optional unique ID for the trial.
            start_time_secs (float): Optional start time in seconds.
        """
        with self.writer.as_default():
            hp.hparams(
                hparams,
                trial_id=trial_id,
                start_time_secs=start_time_secs,
            )
            self.writer.flush()

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


#
# Example usage
#

if __name__ == "__main__":
    logger = TensorboardLogger(experiment_name="jax_demo")

    logger.log_hparams(
        {"lr": 0.01, "batch_size": 64, "optimizer": "adam"},
    )

    # Example plot function
    def plot_sine_wave(data):
        """Userdefined plotter that returns a matplotlib figure."""
        x, y = data
        fig, ax = plt.subplots()
        ax.plot(x, y, color="tab:blue")
        ax.set_title("Sine Wave")
        ax.set_xlabel("x")
        ax.set_ylabel("sin(x)")
        return fig

    for step in range(5):
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x + step / 5.0)
        logger.log_plot("plots/sine_wave", plot_sine_wave, (x, y), step)
        logger.log_scalar("train/loss", np.random.random(), step)
        logger.log_histogram("params/weights", np.random.randn(1000), step)

    logger.close()
