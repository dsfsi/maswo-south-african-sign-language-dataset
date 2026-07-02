# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import codecs
import collections
import json
import math
import os
import sys
import time

import tensorflow as tf


def check_tensorflow_version():
    """Check TensorFlow version."""
    # TF2: Update version check
    major, minor, patch = tf.__version__.split('.')[:3]
    major, minor = int(major), int(minor.split('-')[0])  # Handle versions like "2.x-dev"
    
    if major < 2:
        raise EnvironmentError(
            "TensorFlow version must >= 2.0.0, current version: %s" % tf.__version__
        )


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    print(s, end="", file=sys.stdout)
    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_hparams(hparams, skip_patterns=None):
    """Print hparams, can skip keys based on pattern."""
    values = hparams.values()
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key for skip_pattern in skip_patterns]):
            print_out("  %s=%s" % (key, str(values[key])))


def load_hparams(model_dir):
    """Load hparams from an existing model directory."""
    hparams_file = os.path.join(model_dir, "hparams")
    
    # TF2: tf.gfile → tf.io.gfile
    if tf.io.gfile.exists(hparams_file):
        print_out("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(
                tf.io.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                # TF2: Create HParams object (assuming you're using the custom class)
                # If using tf-hparams package, import accordingly
                # from . import create_hparams_class
                HParams = create_hparams_class()
                hparams = HParams(**hparams_values)
            except ValueError:
                print_out("  can't load hparams file")
                return None
        return hparams
    else:
        return None


def maybe_parse_standard_hparams(hparams, hparams_path):
    """Override hparams values with existing standard hparams config."""
    if not hparams_path:
        return hparams

    # TF2: tf.gfile → tf.io.gfile
    if tf.io.gfile.exists(hparams_path):
        print_out("# Loading standard hparams from %s" % hparams_path)
        with tf.io.gfile.GFile(hparams_path, "r") as f:
            # Parse JSON and update hparams
            hparams_dict = json.load(f)
            for key, value in hparams_dict.items():
                if hasattr(hparams, key):
                    setattr(hparams, key, value)
                else:
                    hparams.add_hparam(key, value)

    return hparams


def save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    print_out("  saving hparams to %s" % hparams_file)
    
    # TF2: tf.gfile → tf.io.gfile
    with codecs.getwriter("utf-8")(
            tf.io.gfile.GFile(hparams_file, "wb")) as f:
        # Convert hparams to JSON
        hparams_dict = hparams.values()
        f.write(json.dumps(hparams_dict, indent=2))


def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor.
    
    Note: tf.Print is deprecated in TF2. Use tf.print instead.
    """
    if not msg:
        msg = s.name if hasattr(s, 'name') else 'tensor'
    
    # TF2: tf.Print → tf.print (different behavior)
    # tf.print is an operation that prints during execution
    tf.print(msg, tf.shape(s), s, summarize=summarize)
    return s


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    # TF2: Summary writing is different
    with summary_writer.as_default():
        tf.summary.scalar(tag, value, step=global_step)


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    """Get TensorFlow configuration protocol.
    
    Note: In TF2 eager mode, ConfigProto is not commonly used.
    GPU memory growth is set differently.
    """
    # TF2: Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # For compatibility, return a config proto for use with tf.compat.v1.Session
    config_proto = tf.compat.v1.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement
    )
    config_proto.gpu_options.allow_growth = True
    return config_proto


def format_text(words):
    """Convert a sequence words into sentence."""
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.abc.Iterable)):  # TF2: collections.Iterable → collections.abc.Iterable
        words = [words]
    return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = b""
    if isinstance(symbols, str):
        symbols = symbols.encode()
    delimiter_len = len(delimiter)
    for symbol in symbols:
        if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
            word += symbol[:-delimiter_len]
        else:  # end of a word
            word += symbol
            words.append(word)
            word = b""
    return b" ".join(words)


def create_hparams_class():
    """Create and return HParams class for backward compatibility.
    
    This is a helper function to create the HParams class that was
    removed from TensorFlow 2.x (was in tf.contrib.training.HParams).
    """
    class HParams:
        """Simple hyperparameters container."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def add_hparam(self, name, value):
            """Add a hyperparameter."""
            setattr(self, name, value)
        
        def values(self):
            """Return all hyperparameters as a dict."""
            return self.__dict__.copy()
        
        def to_json(self):
            """Convert to JSON string."""
            return json.dumps(self.values(), indent=2)
        
        def parse_json(self, json_string):
            """Parse JSON string and update hparams."""
            hparams_dict = json.loads(json_string)
            for key, value in hparams_dict.items():
                setattr(self, key, value)
        
        def __repr__(self):
            return self.to_json()
    
    return HParams