import tensorflow as tf

import attention_model
import model_helper
from utils import misc_utils as utils

__all__ = ["GNMTModel"]


class GNMTModel(attention_model.AttentionModel):
    """Sequence-to-sequence dynamic model with GNMT attention architecture.
    """

    def __init__(self, hparams, mode, iterator,
                 target_vocab_table, reverse_target_vocab_table=None,
                 scope=None, single_cell_fn=None):

        super(GNMTModel, self).__init__(
            hparams=hparams, 
            mode=mode, 
            iterator=iterator,
            target_vocab_table=target_vocab_table,
            reverse_target_vocab_table=reverse_target_vocab_table,
            scope=scope, 
            single_cell_fn=single_cell_fn
        )

    def _build_encoder(self, hparams):
        """Build a GNMT encoder."""
        if hparams.encoder_type == "uni" or hparams.encoder_type == "bi":
            return super(GNMTModel, self)._build_encoder(hparams)

        if hparams.encoder_type != "gnmt":
            raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

        # Build GNMT encoder.
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers
        num_bi_layers = 1
        num_uni_layers = num_layers - num_bi_layers
        utils.print_out("  num_bi_layers = %d" % num_bi_layers)
        utils.print_out("  num_uni_layers = %d" % num_uni_layers)

        iterator = self.iterator
        source = self.cnn.output

        if self.time_major:
            source = tf.transpose(source)

        with tf.name_scope("encoder"):
            dtype = tf.float32

            # Execute _build_bidirectional_rnn from Model class
            bi_encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
                inputs=source,
                sequence_length=iterator.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=0,  # no residual connection
            )

            uni_cell = model_helper.create_rnn_cell(
                unit_type=hparams.unit_type,
                num_units=hparams.num_units,
                num_layers=num_uni_layers,
                num_residual_layers=num_residual_layers,
                forget_bias=hparams.forget_bias,
                dropout=hparams.dropout,
                base_gpu=hparams.base_gpu,
                mode=self.mode,
                single_cell_fn=self.single_cell_fn
            )

            # encoder_outputs: size [max_time, batch_size, num_units]
            #   when time_major = True
            # TF2: Use keras RNN layer or tf.nn.dynamic_rnn (still available in TF2)
            rnn_layer = tf.keras.layers.RNN(
                uni_cell, 
                return_sequences=True, 
                return_state=True,
                time_major=self.time_major
            )
            
            outputs = rnn_layer(
                bi_encoder_outputs,
                mask=tf.sequence_mask(
                    iterator.source_sequence_length,
                    dtype=dtype
                ) if iterator.source_sequence_length is not None else None
            )
            
            encoder_outputs = outputs[0]
            encoder_state = outputs[1:] if len(outputs) > 2 else outputs[1]

            # Pass all encoder state except the first bi-directional layer's state to
            # decoder.
            encoder_state = (bi_encoder_state[1],) + (
                (encoder_state,) if num_uni_layers == 1 else encoder_state
            )

        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, 
                           source_sequence_length):
        """Build a RNN cell with GNMT attention architecture."""

        attention_option = hparams.attention
        attention_architecture = hparams.attention_architecture
        num_units = hparams.num_units
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers
        beam_width = hparams.beam_width

        dtype = tf.float32

        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        # TF2: Use tf.estimator.ModeKeys instead of tf.contrib.learn.ModeKeys
        if self.mode == "predict" and beam_width > 0:
            # TF2: tile_batch moved to tfa or implement manually
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
            
            source_sequence_length = tf.repeat(
                source_sequence_length, 
                repeats=beam_width
            )
            
            # Tile encoder_state
            if isinstance(encoder_state, tuple):
                encoder_state = tuple(
                    tf.repeat(
                        tf.expand_dims(state, 1), 
                        repeats=beam_width, 
                        axis=1
                    ) if not isinstance(state, tuple) else state
                    for state in encoder_state
                )
            
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = attention_model.create_attention_mechanism(
            attention_option, num_units, memory, source_sequence_length
        )

        cell_list = model_helper._cell_list(  # pylint: disable=protected-access
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

        # Only wrap the bottom layer with the attention mechanism.
        attention_cell = cell_list.pop(0)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (
            self.mode == "predict" and beam_width == 0
        )
        
        # TF2: AttentionWrapper is in tfa.seq2seq or tf.keras
        # Using tensorflow_addons if available, otherwise need custom implementation
        try:
            import tensorflow_addons as tfa
            attention_cell = tfa.seq2seq.AttentionWrapper(
                attention_cell,
                attention_mechanism,
                attention_layer_size=None,  # don't use attention layer.
                output_attention=False,
                alignment_history=alignment_history,
                name="attention"
            )
        except ImportError:
            # Fallback: use custom implementation or legacy compat
            attention_cell = tf.compat.v1.nn.rnn_cell.AttentionWrapper(
                attention_cell,
                attention_mechanism,
                attention_layer_size=None,
                output_attention=False,
                alignment_history=alignment_history,
                name="attention"
            )

        if attention_architecture == "gnmt":
            cell = GNMTAttentionMultiCell(attention_cell, cell_list)
        elif attention_architecture == "gnmt_v2":
            cell = GNMTAttentionMultiCell(
                attention_cell, cell_list, use_new_attention=True
            )
        else:
            raise ValueError(
                "Unknown attention_architecture %s" % attention_architecture
            )

        if hparams.pass_hidden_state:
            # Get zero state
            zero_state = cell.get_initial_state(
                batch_size=batch_size, dtype=dtype
            )
            
            # Clone states with encoder state
            decoder_initial_state = tuple(
                _clone_cell_state(zs, es) 
                for zs, es in zip(zero_state, encoder_state)
            )
        else:
            decoder_initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=dtype
            )

        return cell, decoder_initial_state

    def _get_infer_summary(self, hparams):
        if hparams.beam_width > 0:
            return tf.no_op()
        return attention_model._create_attention_images_summary(
            self.final_context_state[0]
        )


def _clone_cell_state(zero_state, encoder_state):
    """Helper function to clone attention wrapper state with encoder state."""
    try:
        # For AttentionWrapperState
        if hasattr(zero_state, 'clone'):
            return zero_state.clone(cell_state=encoder_state)
        else:
            return encoder_state
    except:
        return encoder_state


class GNMTAttentionMultiCell(tf.keras.layers.Layer):
    """A MultiCell with GNMT attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=False):
        """Creates a GNMTAttentionMultiCell.

        Args:
          attention_cell: An instance of AttentionWrapper.
          cells: A list of RNNCell wrapped with AttentionInputWrapper.
          use_new_attention: Whether to use the attention generated from current
            step bottom layer's output. Default is False.
        """
        super(GNMTAttentionMultiCell, self).__init__()
        self._cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention

    @property
    def state_size(self):
        """The state size of the cell."""
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        """The output size of the cell."""
        return self._cells[-1].output_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state for all cells."""
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        
        return tuple(
            cell.get_initial_state(batch_size=batch_size, dtype=dtype)
            for cell in self._cells
        )

    def call(self, inputs, state, training=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not isinstance(state, (list, tuple)):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" % 
                (len(self.state_size), state)
            )

        new_states = []

        # First cell with attention
        with tf.name_scope("cell_0_attention"):
            attention_cell = self._cells[0]
            attention_state = state[0]
            cur_inp, new_attention_state = attention_cell(
                inputs, attention_state, training=training
            )
            new_states.append(new_attention_state)

        # Remaining cells
        for i in range(1, len(self._cells)):
            with tf.name_scope("cell_%d" % i):
                cell = self._cells[i]
                cur_state = state[i]

                # TF2: LSTMStateTuple is replaced with tuple or list
                if not isinstance(cur_state, (tuple, list, tf.keras.layers.LSTMCell)):
                    # For LSTM cells, state should be (h, c) tuple
                    if hasattr(cur_state, 'h') and hasattr(cur_state, 'c'):
                        # Convert named tuple to regular values
                        h = cur_state.h
                        c = cur_state.c
                    else:
                        raise TypeError(
                            "`state[{}]` must be a tuple or LSTMStateTuple".format(i)
                        )
                else:
                    if isinstance(cur_state, (tuple, list)) and len(cur_state) == 2:
                        h, c = cur_state
                    else:
                        h = cur_state
                        c = None

                # Concatenate hidden state with attention
                if self.use_new_attention:
                    attention_context = new_attention_state.attention if hasattr(
                        new_attention_state, 'attention'
                    ) else new_attention_state
                    h = tf.concat([h, attention_context], 1)
                else:
                    attention_context = attention_state.attention if hasattr(
                        attention_state, 'attention'
                    ) else attention_state
                    h = tf.concat([h, attention_context], 1)

                # Reconstruct state
                if c is not None:
                    cur_state = (h, c)
                else:
                    cur_state = h

                cur_inp, new_state = cell(cur_inp, cur_state, training=training)
                new_states.append(new_state)

        return cur_inp, tuple(new_states)
