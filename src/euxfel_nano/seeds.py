"""Seeding, so a rerun reproduces the published numbers.

Which results are deterministic, and how
----------------------------------------
* **Bootstrap uncertainties** (Fig. 3b, Fig. S10) are fully deterministic.
  :func:`euxfel_nano.yields.get_yields` seeds its own generator
  (``numpy.random.default_rng(seed)``, default ``seed=0``), so the published
  error bars are reproduced exactly without any global seeding.

* **Monomer/dimer CNN** (Fig. 3c, 3d, S13) is *not* bit-reproducible from a
  retrain. The train/test split is seeded (``random_state=42``), but GPU
  kernel scheduling and cuDNN autotuning make the fitted weights vary between
  runs even at a fixed seed. Call :func:`set_all_seeds` before training to
  remove every source of randomness that can be controlled; the classification
  reported in the paper comes from the deposited weights, which are the
  authoritative artifact.

* **Synthetic dimer augmentation** draws particle sizes, separations,
  orientations and intensity scalings from distributions matched to the
  experiment. Seed it through :func:`set_all_seeds` for a reproducible
  training set.
"""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_all_seeds(seed=DEFAULT_SEED, deterministic_ops=True):
    """Seed Python, NumPy and, if installed, TensorFlow.

    Parameters
    ----------
    seed
        Seed value. The published classifier used 42.
    deterministic_ops
        Also request deterministic GPU kernels from TensorFlow. This slows
        training but removes run-to-run variation from nondeterministic
        reductions.

    Returns
    -------
    numpy.random.Generator
        A generator seeded from ``seed``, for code that prefers to thread an
        explicit generator rather than rely on global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)
        if deterministic_ops and hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()

    return np.random.default_rng(seed)
