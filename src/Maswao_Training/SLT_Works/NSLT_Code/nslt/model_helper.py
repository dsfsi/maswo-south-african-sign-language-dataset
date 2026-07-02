"""Utility functions for building models."""
import time

import tensorflow as tf

from utils import misc_utils as utils

__all__ = ["get_initializer", "get_device_str", "create_emb_for_encoder_and_decoder", "create_rnn_cell",
           "gradient_clip", "create_or_load_model", "load_model", "compute_perplexity"]

def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.keras.initializers.RandomUniform(minval=-init_weight,
                                                   maxval=init_weight,
                                                   seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.GlorotNormal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.GlorotUniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id):
    """Return a device string for multi-GPU setup."""
    device_str_output = "/GPU:%d" % device_id
    return device_str_output


def create_emb_for_encoder_and_decoder(tgt_vocab_size, tgt_embed_size, dtype=tf.float32, num_partitions=0, scope=None):
    """Create embedding matrix for both encoder and decoder.

    Args:
        tgt_vocab_size: An integer. The target vocab size.
        tgt_embed_size: An integer. The embedding dimension for the decoder's
            embedding.
        dtype: dtype of the embedding matrix. Default to float32.
        num_partitions: number of partitions used for the embedding vars.
        scope: VariableScope for the created subgraph. Default to "embedding".

    Returns:
        embedding_decoder: Decoder's embedding matrix.

    Raises:
        ValueError: if use share_vocab but source and target have different vocab
            size.
    """
    # tf.variable_scope and tf.get_variable are gone in TF2
    # Use tf.keras.layers.Embedding instead
    embedding_decoder = tf.keras.layers.Embedding(
        input_dim=tgt_vocab_size,
        output_dim=tgt_embed_size,
        dtype=dtype,
        name=(scope or "embeddings") + "_decoder"
    )
    return embedding_decoder


def _single_cell(unit_type, num_units, forget_bias, dropout, mode, residual_connection=False, device_str=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == "train" else 0.0

    # Cell Type
    if unit_type == "lstm":
        utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.keras.layers.LSTMCell(num_units)
    elif unit_type == "gru":
        utils.print_out("  GRU", new_line=False)
        single_cell = tf.keras.layers.GRUCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout
    if dropout > 0.0:
        # In TF2, dropout is handled inside the cell itself
        single_cell = tf.keras.layers.LSTMCell(num_units, dropout=dropout) if unit_type == "lstm" else \
                      tf.keras.layers.GRUCell(num_units, dropout=dropout)
        utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout), new_line=False)

    # Residual connection: must be handled manually in the model graph
    if residual_connection:
        utils.print_out("  Residual connection enabled", new_line=False)

    # Device placement: use tf.device context instead of DeviceWrapper
    if device_str:
        utils.print_out("  Device=%s" % device_str, new_line=False)

    return single_cell

def _cell_list(unit_type, num_units, num_layers, num_residual_layers, forget_bias, dropout, mode, base_gpu, single_cell_fn=None):
    """Create a list of RNN cells."""
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    for i in range(num_layers):
        utils.print_out("  cell %d" % i, new_line=False)
        single_cell = single_cell_fn(unit_type=unit_type,
                                     num_units=num_units,
                                     forget_bias=forget_bias,
                                     dropout=dropout,
                                     mode=mode,
                                     residual_connection=(i >= num_layers - num_residual_layers),
                                     device_str=get_device_str(base_gpu))
        utils.print_out("")
        cell_list.append(single_cell)

    return cell_list

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers, forget_bias, dropout, mode, base_gpu, single_cell_fn=None):
    """Create multi-layer RNN cell.

    Args:
        unit_type: string representing the unit type, i.e. "lstm".
        num_units: the depth of each unit.
        num_layers: number of cells.
        num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
            cells in the returned list will be wrapped with `ResidualWrapper`.
        forget_bias: the initial forget bias of the RNNCell(s).
        dropout: floating point value between 0.0 and 1.0:
            the probability of dropout. this is ignored if `mode != TRAIN`.
        mode: either tf.estimator.ModeKeys.TRAIN/EVAL/PREDICT
        base_gpu: The gpu device id to use for the first RNN cell in the
            returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
            as its device id.
        single_cell_fn: allow for adding customized cell.
            When not specified, we default to model_helper._single_cell
    Returns:
        An `RNNCell` instance.
    """
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           base_gpu=base_gpu,
                           single_cell_fn=single_cell_fn)

    if len(cell_list) == 1:  # Single layer.
        return tf.keras.layers.RNN(cell_list[0], return_sequences=True)
    else:  # Multi layers
        return tf.keras.layers.RNN(cell_list, return_sequences=True)

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    tf.summary.scalar("grad_norm", gradient_norm)
    tf.summary.scalar("clipped_gradient", tf.linalg.global_norm(clipped_gradients))
    return clipped_gradients, gradient_norm

def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.load_weights(ckpt)
    utils.print_out("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model

def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters.
    
    Args:
        model: The model instance.
        model_dir: Directory containing checkpoints.
        name: Name for logging.
        
    Returns:
        Tuple of (model, global_step)
    """
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        utils.print_out("  created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))

    global_step = model.optimizer.iterations.numpy()
    return model, global_step


def initialize_cnn(model, session):
    model.cnn.load_initial_weights(session)

def compute_perplexity(model, sess, name):
    """Compute perplexity of the output of the model.

    Args:
        model: model for compute perplexity.
        sess: unused in TF2, kept for compatibility.
        name: name of the batch.

    Returns:
        The perplexity of the eval outputs.
    """
    total_loss = 0.0
    total_predict_count = 0
    start_time = time.time()

    while True:
        try:
            # In TF2, model.eval should be implemented to run eagerly
            # and return (loss, predict_count, batch_size)
            loss, predict_count, batch_size = model.eval()
            total_loss += float(loss) * batch_size
            total_predict_count += int(predict_count)
        except tf.errors.OutOfRangeError:
            break

    perplexity = utils.safe_exp(total_loss / total_predict_count)
    utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                     start_time)
    return perplexity
