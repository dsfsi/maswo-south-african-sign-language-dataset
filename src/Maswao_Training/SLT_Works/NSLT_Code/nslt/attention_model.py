import tensorflow as tf

import model
import model_helper

__all__ = ["AttentionModel"]


class AttentionModel(model.Model):
    """Sequence-to-sequence dynamic model with attention.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    (Luong et al., EMNLP'2015) paper: https://arxiv.org/pdf/1508.04025v5.pdf.
    This class also allows to use GRU cells in addition to LSTM cells with
    support for dropout.
    """

    def __init__(self, hparams, mode, iterator,
                 target_vocab_table, reverse_target_vocab_table=None,
                 scope=None, single_cell_fn=None):

        super(AttentionModel, self).__init__(
            hparams=hparams, 
            mode=mode, 
            iterator=iterator,
            target_vocab_table=target_vocab_table,
            reverse_target_vocab_table=reverse_target_vocab_table,
            scope=scope, 
            single_cell_fn=single_cell_fn
        )

        if self.mode == "predict":
            self.infer_summary = self._get_infer_summary(hparams)

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, 
                           source_sequence_length):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        attention_option = hparams.attention
        attention_architecture = hparams.attention_architecture

        if attention_architecture != "standard":
            raise ValueError(
                "Unknown attention architecture %s" % attention_architecture
            )

        num_units = hparams.num_units
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers
        beam_width = hparams.beam_width

        dtype = tf.float32

        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        # TF2: tile_batch implementation for beam search
        if self.mode == "predict" and beam_width > 0:
            # Tile memory
            memory = tf.repeat(
                tf.expand_dims(memory, 1), 
                repeats=beam_width, 
                axis=1
            )
            memory = tf.reshape(
                memory,
                [tf.shape(memory)[0] * beam_width,
                 tf.shape(memory)[2],
                 tf.shape(memory)[3]]
            )
            
            # Tile source_sequence_length
            source_sequence_length = tf.repeat(
                source_sequence_length, 
                repeats=beam_width
            )
            
            # Tile encoder_state
            if isinstance(encoder_state, tuple):
                encoder_state = tuple(
                    _tile_state(state, beam_width) 
                    for state in encoder_state
                )
            else:
                encoder_state = _tile_state(encoder_state, beam_width)
            
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = create_attention_mechanism(
            attention_option, num_units, memory, source_sequence_length
        )

        cell = model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            base_gpu=hparams.base_gpu,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn
        )

        # Only generate alignment in greedy INFER mode.
        alignment_history = (
            self.mode == "predict" and beam_width == 0
        )

        # TF2: Use tensorflow_addons for AttentionWrapper
        try:
            import tensorflow_addons as tfa
            cell = tfa.seq2seq.AttentionWrapper(
                cell, 
                attention_mechanism, 
                attention_layer_size=num_units,
                alignment_history=alignment_history, 
                name="attention"
            )
        except ImportError:
            # Fallback to compat.v1
            cell = tf.compat.v1.nn.rnn_cell.AttentionWrapper(
                cell, 
                attention_mechanism, 
                attention_layer_size=num_units,
                alignment_history=alignment_history, 
                name="attention"
            )

        # TF2: DeviceWrapper is not needed in eager mode
        # Device placement is handled by tf.device context manager
        with tf.device(model_helper.get_device_str(hparams.base_gpu)):
            if hparams.pass_hidden_state:
                # Get initial state
                decoder_initial_state = cell.get_initial_state(
                    batch_size=batch_size, 
                    dtype=dtype
                )
                # Clone with encoder state
                if hasattr(decoder_initial_state, 'clone'):
                    decoder_initial_state = decoder_initial_state.clone(
                        cell_state=encoder_state
                    )
                else:
                    # For custom implementations, manually set cell_state
                    decoder_initial_state = _clone_attention_state(
                        decoder_initial_state, encoder_state
                    )
            else:
                decoder_initial_state = cell.get_initial_state(
                    batch_size=batch_size, 
                    dtype=dtype
                )

        return cell, decoder_initial_state

    def _get_infer_summary(self, hparams):
        if hparams.beam_width > 0:
            return tf.no_op()
        return _create_attention_images_summary(self.final_context_state)


def _tile_state(state, beam_width):
    """Helper function to tile state for beam search."""
    if isinstance(state, tuple):
        # Handle LSTM state (c, h)
        return tuple(_tile_state(s, beam_width) for s in state)
    else:
        # Regular tensor
        return tf.repeat(
            tf.expand_dims(state, 1), 
            repeats=beam_width, 
            axis=1
        )


def _clone_attention_state(zero_state, encoder_state):
    """Helper to clone attention wrapper state with encoder state."""
    # This is a simplified version - may need adjustment based on actual state structure
    if hasattr(zero_state, '_replace'):
        # Named tuple
        return zero_state._replace(cell_state=encoder_state)
    elif isinstance(zero_state, dict):
        # Dictionary
        new_state = zero_state.copy()
        new_state['cell_state'] = encoder_state
        return new_state
    else:
        # Fallback: return encoder_state
        return encoder_state


def create_attention_mechanism(attention_option, num_units, memory, 
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    # TF2: Use tensorflow_addons for attention mechanisms
    try:
        import tensorflow_addons as tfa
        
        if attention_option == "luong":
            attention_mechanism = tfa.seq2seq.LuongAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length
            )
        elif attention_option == "scaled_luong":
            attention_mechanism = tfa.seq2seq.LuongAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length, 
                scale=True
            )
        elif attention_option == "bahdanau":
            attention_mechanism = tfa.seq2seq.BahdanauAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length
            )
        elif attention_option == "normed_bahdanau":
            attention_mechanism = tfa.seq2seq.BahdanauAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length, 
                normalize=True
            )
        else:
            raise ValueError("Unknown attention option %s" % attention_option)
            
    except ImportError:
        # Fallback to compat.v1
        if attention_option == "luong":
            attention_mechanism = tf.compat.v1.nn.seq2seq.LuongAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length
            )
        elif attention_option == "scaled_luong":
            attention_mechanism = tf.compat.v1.nn.seq2seq.LuongAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length, 
                scale=True
            )
        elif attention_option == "bahdanau":
            attention_mechanism = tf.compat.v1.nn.seq2seq.BahdanauAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length
            )
        elif attention_option == "normed_bahdanau":
            attention_mechanism = tf.compat.v1.nn.seq2seq.BahdanauAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length, 
                normalize=True
            )
        else:
            raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def _create_attention_images_summary(final_context_state):
    """Create attention image and attention summary."""
    # Extract alignment history
    if hasattr(final_context_state, 'alignment_history'):
        alignment_history = final_context_state.alignment_history
        if hasattr(alignment_history, 'stack'):
            attention_images = alignment_history.stack()
        else:
            # If alignment_history is already a tensor
            attention_images = alignment_history
    else:
        # No alignment history available
        return tf.no_op()
    
    # Reshape to (batch, src_seq_len, tgt_seq_len, 1)
    attention_images = tf.expand_dims(
        tf.transpose(attention_images, [1, 2, 0]), 
        -1
    )
    
    # Scale to range [0, 255]
    attention_images = attention_images * 255
    
    # TF2: Create summary using tf.summary.image
    # Note: This should be called within a summary writer context
    with tf.name_scope('attention_summary'):
        # Convert to uint8 for image summary
        attention_images = tf.cast(attention_images, tf.uint8)
        
        # In TF2, summaries are written differently
        # This returns the summary operation
        return tf.summary.image("attention_images", attention_images, max_outputs=3)