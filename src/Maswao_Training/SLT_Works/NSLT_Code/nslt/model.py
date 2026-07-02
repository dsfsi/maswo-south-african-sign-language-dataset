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

"""Basic sequence-to-sequence model with dynamic RNN support."""

import abc
import tensorflow as tf

import model_helper
from utils import iterator_utils
from utils import misc_utils as utils
from alexnet import AlexNet

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class BaseModel(object):
    """Sequence-to-sequence base class.
    """

    def __init__(self, hparams, mode, iterator, target_vocab_table,
                 reverse_target_vocab_table=None, scope=None, single_cell_fn=None):
        """Create the model.

        Args:
          hparams: Hyperparameter configurations.
          mode: TRAIN | EVAL | PREDICT
          iterator: Dataset Iterator that feeds data.
          target_vocab_table: Lookup table mapping target words to ids.
          reverse_target_vocab_table: Lookup table mapping ids to target words. Only
            required in PREDICT mode. Defaults to None.
          scope: scope of the model.
          single_cell_fn: allow for adding customized cell. When not specified,
            we default to model_helper._single_cell
        """
        assert isinstance(iterator, iterator_utils.BatchedInput)

        self.iterator = iterator
        self.mode = mode
        self.tgt_vocab_table = target_vocab_table

        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.num_layers = hparams.num_layers
        self.num_gpus = hparams.num_gpus
        self.time_major = hparams.time_major

        self.cnn_input = self.iterator.source
        if self.mode == "train":
            self.cnn = AlexNet(
                self.cnn_input,
                (1 - hparams.dropout),
                model_helper.get_device_str(hparams.base_gpu)
            )
        else:
            self.cnn = AlexNet(
                self.cnn_input,
                1,
                model_helper.get_device_str(hparams.base_gpu)
            )

        # Initializer
        initializer = model_helper.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight
        )
        tf.keras.backend.set_floatx('float32')

        # Embeddings
        self.init_embeddings(hparams, scope)

        # Batch size
        self.batch_size = tf.shape(self.iterator.source_sequence_length)[0]

        # Projection
        with tf.name_scope(scope or "build_network"):
            with tf.name_scope("decoder/output_projection"):
                self.output_layer = tf.keras.layers.Dense(
                    hparams.tgt_vocab_size,
                    use_bias=False,
                    name="output_projection",
                    kernel_initializer=initializer
                )

        # To make it flexible for external code to add other cell types
        self.single_cell_fn = single_cell_fn

        # Build graph
        res = self.build_graph(hparams, scope=scope)

        if self.mode == "train":
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(self.iterator.target_sequence_length)
        elif self.mode == "eval":
            self.eval_loss = res[1]
        elif self.mode == "predict":
            self.infer_logits, _, self.final_context_state, self.sample_id = res
            self.sample_words = reverse_target_vocab_table.lookup(
                tf.cast(self.sample_id, tf.int64)
            )

        if self.mode != "predict":
            # Count predicted words for ppl
            self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)

        # Learning rate
        print("  start_decay_step=%d, learning_rate=%g, decay_steps %d, decay_factor %g" %
              (hparams.start_decay_step, hparams.learning_rate,
               hparams.decay_steps, hparams.decay_factor))

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Get trainable variables
        self.trainable_vars = []
        for layer in [self.cnn, self.output_layer]:
            if hasattr(layer, 'trainable_variables'):
                self.trainable_vars.extend(layer.trainable_variables)

        # Optimizer (train mode)
        if self.mode == "train":
            if hparams.optimizer == "sgd":
                self.learning_rate = tf.Variable(
                    hparams.learning_rate,
                    trainable=False,
                    dtype=tf.float32
                )
                self.opt = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate
                )
            elif hparams.optimizer == "adam":
                assert float(hparams.learning_rate) <= 0.001, \
                    "! High Adam learning rate %g" % hparams.learning_rate
                self.learning_rate = tf.Variable(
                    hparams.learning_rate,
                    trainable=False,
                    dtype=tf.float32
                )
                self.opt = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate
                )
            self.hparams = hparams

        # Summary writer
        self.summary_writer = None

        # Print trainable variables
        utils.print_out("# Trainable variables")
        for var in self.trainable_vars:
            utils.print_out("  %s, %s" % (var.name, str(var.shape)))

    def init_embeddings(self, hparams, scope):
        """Init embeddings."""
        self.embedding_decoder = tf.keras.layers.Embedding(
            input_dim=self.tgt_vocab_size,
            output_dim=hparams.num_units,
            name=(scope or "embeddings") + "_decoder"
        )

    @tf.function
    def train_step(self):
        """Execute one training step."""
        assert self.mode == "train"

        with tf.GradientTape() as tape:
            # Forward pass is already done in build_graph
            loss = self.train_loss

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_vars)

        # Clip gradients
        clipped_gradients, gradient_norm = model_helper.gradient_clip(
            gradients,
            max_gradient_norm=self.hparams.max_gradient_norm
        )

        # Apply gradients
        self.opt.apply_gradients(zip(clipped_gradients, self.trainable_vars))

        # Update global step
        self.global_step.assign_add(1)

        # Update learning rate if using SGD with decay
        if self.hparams.optimizer == "sgd":
            if self.global_step >= self.hparams.start_decay_step:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=self.hparams.learning_rate,
                    decay_steps=self.hparams.decay_steps,
                    decay_rate=self.hparams.decay_factor,
                    staircase=True
                )
                decayed_lr = lr_schedule(self.global_step - self.hparams.start_decay_step)
                self.learning_rate.assign(decayed_lr)

        return (None, loss, self.predict_count,
                None, self.global_step, self.word_count, self.batch_size)

    def train(self):
        """Wrapper for training step in TF2."""
        return self.train_step()

    def eval(self):
        """Evaluate the model."""
        assert self.mode == "eval"
        return [self.eval_loss, self.predict_count, self.batch_size]

    def build_graph(self, hparams, scope=None):
        """Subclass must implement this method.

        Creates a sequence-to-sequence model with dynamic RNN decoder API.
        """
        utils.print_out("# creating %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.name_scope(scope or "dynamic_seq2seq"):
            # Encoder
            encoder_outputs, encoder_state = self._build_encoder(hparams)

            # Decoder
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, encoder_state, hparams
            )

            # Loss
            if self.mode != "predict":
                with tf.device(model_helper.get_device_str(hparams.base_gpu)):
                    loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

    @abc.abstractmethod
    def _build_encoder(self, hparams):
        """Subclass must implement this."""
        pass

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers):
        """Build a multi-layer RNN cell that can be used by encoder."""
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            mode=self.mode,
            base_gpu=hparams.base_gpu,
            single_cell_fn=self.single_cell_fn
        )

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        static = tensor.shape[time_axis]
        return static if static is not None else tf.shape(tensor)[time_axis]

    @abc.abstractmethod
    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Subclass must implement this."""
        pass

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.iterator.target_output

        if self.time_major:
            target_output = tf.transpose(target_output)

        max_time = self.get_max_time(target_output)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits
        )

        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length,
            max_time
        )

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(crossent * tf.cast(target_weights, logits.dtype)) / tf.cast(
            self.batch_size, tf.float32
        )

        return loss

    def _get_infer_summary(self, hparams):
        return None

    def infer(self):
        """Run inference."""
        assert self.mode == "predict"
        return [self.infer_logits, self._get_infer_summary(self.hparams),
                self.sample_id, self.sample_words]

    def decode(self):
        """Decode a batch.

        Returns:
          A tuple consisting of outputs, infer_summary.
            outputs: of size [batch_size, time]
        """
        _, infer_summary, _, sample_words = self.infer()

        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            sample_words = tf.transpose(sample_words)
        return sample_words, infer_summary


class Model(BaseModel):
    """Sequence-to-sequence dynamic model.

    This class implements a multi-layer recurrent neural network as encoder,
    and a multi-layer recurrent neural network decoder.
    """

    def _build_encoder(self, hparams):
        """Build an encoder."""
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers

        iterator = self.iterator
        output = self.cnn.output  # Expecting [batch, time, features] or [batch, features]

        with tf.name_scope("encoder"):
            dtype = tf.float32

            # Ensure input to RNN is 3D: [batch, time, features]
            # If CNN output is 2D [batch, features], add a dummy time dim.
            if tf.rank(output) == 2:
                output = tf.expand_dims(output, axis=1)

            if hparams.encoder_type == "uni":
                utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                                (num_layers, num_residual_layers))
                cell = self._build_encoder_cell(
                    hparams, num_layers, num_residual_layers
                )

                rnn = tf.keras.layers.RNN(
                    cell,
                    return_sequences=True,
                    return_state=True,
                )

                result = rnn(
                    output,
                    mask=tf.sequence_mask(
                        iterator.source_sequence_length
                    ) if iterator.source_sequence_length is not None else None
                )

                encoder_outputs = result[0]
                encoder_state = result[1:] if len(result) > 2 else result[1]

            elif hparams.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)
                num_bi_residual_layers = int(num_residual_layers / 2)
                utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                                (num_bi_layers, num_bi_residual_layers))

                encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
                    inputs=output,
                    sequence_length=iterator.source_sequence_length,
                    dtype=dtype,
                    hparams=hparams,
                    num_bi_layers=num_bi_layers,
                    num_bi_residual_layers=num_bi_residual_layers
                )

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

        return encoder_outputs, encoder_state

    def _build_bidirectional_rnn(self, inputs, sequence_length, dtype, hparams,
                                 num_bi_layers, num_bi_residual_layers):
        """Create and call bidirectional RNN cells."""
        # Ensure inputs are 3D
        if tf.rank(inputs) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        fw_cell = self._build_encoder_cell(
            hparams, num_bi_layers, num_bi_residual_layers
        )
        bw_cell = self._build_encoder_cell(
            hparams, num_bi_layers, num_bi_residual_layers
        )

        bi_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(
                fw_cell,
                return_sequences=True,
                return_state=True,
            ),
            backward_layer=tf.keras.layers.RNN(
                bw_cell,
                return_sequences=True,
                return_state=True,
            ),
            merge_mode='concat'
        )

        result = bi_rnn(
            inputs,
            mask=tf.sequence_mask(sequence_length) if sequence_length is not None else None
        )

        bi_outputs = result[0]
        states = result[1:]

        # Split forward/backward states (GRU or LSTM supported)
        fw_states = states[::2] if len(states) > 1 else [states[0]]
        bw_states = states[1::2] if len(states) > 1 else [states[0]]

        bi_state = (tuple(fw_states), tuple(bw_states))

        return bi_outputs, bi_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build an RNN cell that can be used by decoder."""
        if hparams.attention:
            raise ValueError("BasicModel doesn't support attention.")

        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers

        cell = model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            base_gpu=hparams.base_gpu,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn
        )

        # Beam search tiling
        if self.mode == "predict" and hparams.beam_width > 0:
            try:
                import tensorflow_addons as tfa
                decoder_initial_state = tfa.seq2seq.tile_batch(
                    encoder_state, multiplier=hparams.beam_width
                )
            except ImportError:
                if isinstance(encoder_state, tuple):
                    decoder_initial_state = tuple(
                        tf.repeat(tf.expand_dims(s, 1), repeats=hparams.beam_width, axis=1)
                        if not isinstance(s, tuple) else s
                        for s in encoder_state
                    )
                else:
                    decoder_initial_state = tf.repeat(
                        tf.expand_dims(encoder_state, 1), repeats=hparams.beam_width, axis=1
                    )
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        """Build and run a RNN decoder with a final projection layer.

        Args:
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          hparams: The Hyperparameters configurations.

        Returns:
          A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """
        tgt_sos_id = tf.cast(
            self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
            tf.int32
        )
        tgt_eos_id = tf.cast(
            self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
            tf.int32
        )

        iterator = self.iterator

        # maximum_iteration: The maximum decoding steps.
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
            utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            # decoding_length_factor flag could be added later
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(iterator.source_sequence_length)
            maximum_iterations = tf.cast(
                tf.round(tf.cast(max_encoder_length, tf.float32) * decoding_length_factor),
                tf.int32
            )

        ## Decoder.
        with tf.name_scope("decoder"):
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state,
                iterator.source_sequence_length
            )

            ## Train or eval
            if self.mode != "predict":
                # decoder_emb_inp: [batch_size, max_time, num_units] (batch-major)
                target_input = iterator.target_input

                if self.time_major:
                    target_input = tf.transpose(target_input)

                # Keras Embedding lookup
                decoder_emb_inp = self.embedding_decoder(target_input)

                # Training helper/decoder via TF Addons
                import tensorflow_addons as tfa
                helper = tfa.seq2seq.TrainingHelper(
                    decoder_emb_inp,
                    iterator.target_sequence_length,
                    time_major=False  # batch-major in TF2/Keras
                )

                my_decoder = tfa.seq2seq.BasicDecoder(
                    cell, helper, decoder_initial_state
                )

                # Dynamic decoding
                outputs, final_context_state, _ = tfa.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=None
                )

                sample_id = outputs.sample_id

                # Apply projection to all timesteps
                with tf.device(model_helper.get_device_str(hparams.base_gpu)):
                    logits = self.output_layer(outputs.rnn_output)

            ## Inference
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                import tensorflow_addons as tfa

                if beam_width > 0:
                    my_decoder = tfa.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight
                    )
                else:
                    # Greedy helper
                    helper = tfa.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_decoder, start_tokens, end_token
                    )

                    # Basic decoder with output layer per timestep
                    my_decoder = tfa.seq2seq.BasicDecoder(
                        cell, helper, decoder_initial_state,
                        output_layer=self.output_layer
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tfa.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations
                )

                if beam_width > 0:
                    logits = None
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state