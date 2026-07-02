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
import collections
import math
import os
import random
import time

import tensorflow as tf

import attention_model
import gnmt_model
import inference
import model as nmt_model
import model_helper
from utils import iterator_utils
from utils import misc_utils as utils
from utils import nmt_utils
from utils import vocab_utils


utils.check_tensorflow_version()

__all__ = ["create_train_model", "create_eval_model", "run_sample_decode", 
           "run_internal_eval", "run_external_eval", "run_full_eval", "train"]


class TrainModel(collections.namedtuple("TrainModel", 
                 ("graph", "model", "iterator", "skip_count_placeholder"))):
    pass


def create_train_model(model_creator, hparams, scope=None, single_cell_fn=None):
    """Create train graph, model, and iterator."""
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)

    tgt_vocab_file = hparams.tgt_vocab_file

    # TF2: No explicit graph needed in eager mode, but we'll use tf.function
    tgt_vocab_table = vocab_utils.create_tgt_vocab_table(tgt_vocab_file)

    # TF2: Use tf.data.TextLineDataset instead of contrib
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    
    # TF2: Variables are created directly, no placeholder needed in eager mode
    skip_count_placeholder = tf.Variable(0, dtype=tf.int64, trainable=False)

    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        tgt_vocab_table,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder
    )

    # TF2: Device placement using tf.device context
    with tf.device(model_helper.get_device_str(hparams.base_gpu)):
        model = model_creator(
            hparams,
            iterator=iterator,
            mode="train",  # Changed from contrib.learn
            target_vocab_table=tgt_vocab_table,
            scope=scope,
            single_cell_fn=single_cell_fn
        )

    return TrainModel(
        graph=None,  # TF2: No graph in eager mode
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder
    )


class EvalModel(collections.namedtuple("EvalModel", 
                ("graph", "model", "src_file_placeholder", 
                 "tgt_file_placeholder", "iterator"))):
    pass


def create_eval_model(model_creator, hparams, scope=None, single_cell_fn=None):
    """Create eval graph, model, src/tgt file holders, and iterator."""
    tgt_vocab_file = hparams.tgt_vocab_file

    tgt_vocab_table = vocab_utils.create_tgt_vocab_table(tgt_vocab_file)

    # TF2: Use variables instead of placeholders
    src_file_placeholder = tf.Variable("", dtype=tf.string, trainable=False)
    tgt_file_placeholder = tf.Variable("", dtype=tf.string, trainable=False)
    
    src_dataset = tf.data.TextLineDataset(src_file_placeholder)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
    
    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        tgt_vocab_table,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len
    )

    model = model_creator(
        hparams,
        iterator=iterator,
        mode="eval",
        target_vocab_table=tgt_vocab_table,
        scope=scope,
        single_cell_fn=single_cell_fn
    )

    return EvalModel(
        graph=None,
        model=model,
        src_file_placeholder=src_file_placeholder,
        tgt_file_placeholder=tgt_file_placeholder,
        iterator=iterator
    )


def run_sample_decode(infer_model, model_dir, hparams, summary_writer, 
                     src_data, tgt_data):
    """Sample decode a random sentence from src_data."""
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, "infer"
    )

    _sample_decode(
        loaded_infer_model, global_step, hparams, 
        infer_model.iterator, src_data, tgt_data, 
        infer_model.src_placeholder, summary_writer
    )


def run_internal_eval(eval_model, model_dir, hparams, summary_writer):
    """Compute internal evaluation (perplexity) for both dev / test."""
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, "eval"
    )

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    
    dev_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: dev_src_file,
        eval_model.tgt_file_placeholder: dev_tgt_file
    }

    dev_ppl = _internal_eval(
        loaded_eval_model, global_step, eval_model.iterator, 
        dev_eval_iterator_feed_dict, summary_writer, "dev"
    )

    test_ppl = None
    if hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        
        test_eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: test_src_file,
            eval_model.tgt_file_placeholder: test_tgt_file
        }

        test_ppl = _internal_eval(
            loaded_eval_model, global_step, eval_model.iterator, 
            test_eval_iterator_feed_dict, summary_writer, "test"
        )
    
    return dev_ppl, test_ppl


def run_external_eval(infer_model, model_dir, hparams, summary_writer, 
                     save_best_dev=True):
    """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, "infer"
    )

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)

    dev_infer_iterator_feed_dict = {
        infer_model.src_placeholder: inference.load_data(dev_src_file)
    }

    dev_scores = _external_eval(
        loaded_infer_model, global_step, hparams, 
        infer_model.iterator, dev_infer_iterator_feed_dict, 
        dev_tgt_file, "dev", summary_writer, save_on_best=save_best_dev
    )

    test_scores = None
    if hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)

        test_infer_iterator_feed_dict = {
            infer_model.src_placeholder: inference.load_data(test_src_file)
        }

        test_scores = _external_eval(
            loaded_infer_model, global_step, hparams, 
            infer_model.iterator, test_infer_iterator_feed_dict, 
            test_tgt_file, "test", summary_writer, save_on_best=False
        )

    return dev_scores, test_scores, global_step


def run_full_eval(model_dir, infer_model, eval_model, hparams, 
                 summary_writer, sample_src_data, sample_tgt_data):
    """Wrapper for running sample_decode, internal_eval and external_eval."""
    run_sample_decode(
        infer_model, model_dir, hparams, summary_writer, 
        sample_src_data, sample_tgt_data
    )
    
    dev_ppl, test_ppl = run_internal_eval(
        eval_model, model_dir, hparams, summary_writer
    )
    
    dev_scores, test_scores, global_step = run_external_eval(
        infer_model, model_dir, hparams, summary_writer
    )
    
    result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)

    if hparams.test_prefix:
        result_summary += ", " + _format_results(
            "test", test_ppl, test_scores, hparams.metrics
        )

    return result_summary, global_step, dev_scores, test_scores, dev_ppl, test_ppl


def train(hparams, scope=None, target_session="", single_cell_fn=None):
    """Train a translation model."""
    log_device_placement = hparams.log_device_placement
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats

    if hparams.eval_on_fly:
        steps_per_external_eval = hparams.steps_per_external_eval
        steps_per_eval = 10 * steps_per_stats

        if not steps_per_external_eval:
            steps_per_external_eval = 2 * steps_per_eval
    else:
        steps_per_snapshot = hparams.snapshot_interval

    if not hparams.attention:
        model_creator = nmt_model.Model
    elif hparams.attention_architecture == "standard":
        model_creator = attention_model.AttentionModel
    elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
        model_creator = gnmt_model.GNMTModel
    else:
        raise ValueError("Unknown model architecture")

    train_model = create_train_model(model_creator, hparams, scope, single_cell_fn)

    if hparams.eval_on_fly:
        eval_model = create_eval_model(model_creator, hparams, scope, single_cell_fn)
        infer_model = inference.create_infer_model(
            model_creator, hparams, scope, single_cell_fn
        )

        # Preload data for sample decoding.
        dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
        dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
        sample_src_data = inference.load_data(dev_src_file)
        sample_tgt_data = inference.load_data(dev_tgt_file)

    summary_name = "train_log"
    model_dir = hparams.out_dir

    # Log and output files
    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.io.gfile.GFile(log_file, mode="a")  # TF2: tf.io.gfile
    utils.print_out("# log_file=%s" % log_file, log_f)

    avg_step_time = 0.0

    # TF2: Session configuration is different
    # In TF2 eager mode, we don't need sessions, but if using tf.compat.v1:
    # config_proto = tf.compat.v1.ConfigProto(
    #     log_device_placement=log_device_placement
    # )
    
    # TF2: Initialize model
    model_helper.initialize_cnn(train_model.model)
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, "train"
    )

    # Summary writer
    summary_writer = tf.summary.create_file_writer(
        os.path.join(out_dir, summary_name)
    )

    # First evaluation
    if hparams.eval_on_fly:
        run_full_eval(
            model_dir, infer_model, eval_model, 
            hparams, summary_writer, sample_src_data, sample_tgt_data
        )

    last_stats_step = global_step

    if hparams.eval_on_fly:
        last_eval_step = global_step
        last_external_eval_step = global_step
    else:
        last_snapshot_step = global_step

    # This is the training loop.
    step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
    checkpoint_total_count = 0.0
    speed, train_ppl = 0.0, 0.0
    start_train_time = time.time()

    utils.print_out(
        "# Start step %d, lr %g, %s" % (
            global_step, 
            loaded_train_model.learning_rate.numpy(),  # TF2: .numpy() instead of .eval()
            time.ctime()
        ), 
        log_f
    )

    # Initialize all of the iterators
    skip_count = hparams.batch_size * hparams.epoch_step
    utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
    
    # TF2: Assign variable instead of feed_dict
    train_model.skip_count_placeholder.assign(skip_count)

    while global_step < num_train_steps:
        ### Run a step ###
        start_time = time.time()
        try:
            step_result = loaded_train_model.train()
            (_, step_loss, step_predict_count, step_summary, 
             global_step, step_word_count, batch_size) = step_result
            hparams.epoch_step += 1
        except (tf.errors.OutOfRangeError, StopIteration):
            # Finished going through the training dataset. Go to next epoch.
            hparams.epoch_step = 0
            utils.print_out(
                "# Finished an epoch, step %d. Perform external evaluation" % global_step
            )

            if hparams.eval_on_fly:
                run_sample_decode(
                    infer_model, model_dir, hparams, 
                    summary_writer, sample_src_data, sample_tgt_data
                )
                dev_scores, test_scores, _ = run_external_eval(
                    infer_model, model_dir, hparams, summary_writer
                )

            train_model.skip_count_placeholder.assign(0)
            continue

        # Write step summary.
        with summary_writer.as_default():
            tf.summary.experimental.write_raw_pb(step_summary, step=global_step)

        # Update statistics
        step_time += (time.time() - start_time)

        checkpoint_loss += (step_loss * batch_size)
        checkpoint_predict_count += step_predict_count
        checkpoint_total_count += float(step_word_count)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step

            # Print statistics for the previous epoch.
            avg_step_time = step_time / steps_per_stats
            train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
            speed = checkpoint_total_count / (1000 * step_time)
            utils.print_out(
                "  global step %d lr %g step-time %.2fs wps %.2fK ppl %.2f %s" % (
                    global_step, 
                    loaded_train_model.learning_rate.numpy(),
                    avg_step_time, 
                    speed, 
                    train_ppl, 
                    _get_best_results(hparams)
                ), 
                log_f
            )
            if math.isnan(train_ppl):
                break

            # Reset timer and loss.
            step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
            checkpoint_total_count = 0.0

        ##
        if (not hparams.eval_on_fly) and (global_step - last_snapshot_step >= steps_per_snapshot):
            last_snapshot_step = global_step
            utils.print_out("# Cihan: Saving Snapshot, global step %d" % global_step)
            utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

            # Save checkpoint
            loaded_train_model.save_weights(
                os.path.join(out_dir, "translate.ckpt")
            )

        if hparams.eval_on_fly and (global_step - last_eval_step >= steps_per_eval):
            last_eval_step = global_step

            utils.print_out("# Save eval, global step %d" % global_step)
            utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

            # Save checkpoint
            loaded_train_model.save_weights(
                os.path.join(out_dir, "translate.ckpt")
            )

            # Evaluate on dev/test
            run_sample_decode(
                infer_model, model_dir, hparams, 
                summary_writer, sample_src_data, sample_tgt_data
            )
            dev_ppl, test_ppl = run_internal_eval(
                eval_model, model_dir, hparams, summary_writer
            )

        if hparams.eval_on_fly and (global_step - last_external_eval_step >= steps_per_external_eval):
            last_external_eval_step = global_step

            # Save checkpoint
            loaded_train_model.save_weights(
                os.path.join(out_dir, "translate.ckpt")
            )
            run_sample_decode(
                infer_model, model_dir, hparams, 
                summary_writer, sample_src_data, sample_tgt_data
            )
            dev_scores, test_scores, _ = run_external_eval(
                infer_model, model_dir, hparams, summary_writer
            )

    # Done training
    loaded_train_model.save_weights(
        os.path.join(out_dir, "translate.ckpt")
    )

    if hparams.eval_on_fly:
        result_summary, _, dev_scores, test_scores, dev_ppl, test_ppl = run_full_eval(
            model_dir, infer_model, eval_model, 
            hparams, summary_writer, sample_src_data, sample_tgt_data
        )

        utils.print_out(
            "# Final, step %d lr %g step-time %.2f wps %.2fK ppl %.2f, %s, %s" % (
                global_step, 
                loaded_train_model.learning_rate.numpy(),
                avg_step_time, 
                speed, 
                train_ppl, 
                result_summary, 
                time.ctime()
            ), 
            log_f
        )
        utils.print_time("# Done training!", start_train_time)

        utils.print_out("# Start evaluating saved best models.")
        for metric in hparams.metrics:
            best_model_dir = getattr(hparams, "best_" + metric + "_dir")
            result_summary, best_global_step, _, _, _, _ = run_full_eval(
                best_model_dir, infer_model, eval_model, 
                hparams, summary_writer, sample_src_data, sample_tgt_data
            )
            utils.print_out(
                "# Best %s, step %d step-time %.2f wps %.2fK, %s, %s" % (
                    metric, 
                    best_global_step, 
                    avg_step_time, 
                    speed, 
                    result_summary, 
                    time.ctime()
                ), 
                log_f
            )

    summary_writer.close()

    if hparams.eval_on_fly:
        return dev_scores, test_scores, dev_ppl, test_ppl, global_step
    else:
        return global_step


def _format_results(name, ppl, scores, metrics):
    """Format results."""
    result_str = "%s ppl %.2f" % (name, ppl)
    if scores:
        for metric in metrics:
            result_str += ", %s %s %.1f" % (name, metric, scores[metric])
    return result_str


def _get_best_results(hparams):
    """Summary of the current best results."""
    tokens = []
    for metric in hparams.metrics:
        tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
    return ", ".join(tokens)


def _internal_eval(model, global_step, iterator, iterator_feed_dict, 
                  summary_writer, label):
    """Computing perplexity."""
    # TF2: Set variables instead of feed_dict
    for key, value in iterator_feed_dict.items():
        key.assign(value)
    
    ppl = model_helper.compute_perplexity(model, label)
    utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
    return ppl


def _sample_decode(model, global_step, hparams, iterator, src_data, 
                  tgt_data, iterator_src_placeholder, summary_writer):
    """Pick a sentence and decode."""
    decode_id = random.randint(0, len(src_data) - 1)
    utils.print_out("  # %d" % decode_id)

    # TF2: Assign instead of feed_dict
    iterator_src_placeholder.assign([src_data[decode_id]])

    nmt_outputs, attention_summary = model.decode()

    if hparams.beam_width > 0:
        # Get the top translation.
        nmt_outputs = nmt_outputs[0]

    translation = nmt_utils.get_translation(
        nmt_outputs, 
        sent_id=0, 
        tgt_eos=hparams.eos, 
        bpe_delimiter=hparams.bpe_delimiter
    )
    utils.print_out("    src: %s" % src_data[decode_id])
    utils.print_out("    ref: %s" % tgt_data[decode_id])
    utils.print_out("    nmt: %s" % translation)

    # Summary
    if attention_summary is not None:
        with summary_writer.as_default():
            tf.summary.experimental.write_raw_pb(attention_summary, step=global_step)


def _external_eval(model, global_step, hparams, iterator, 
                  iterator_feed_dict, tgt_file, label, 
                  summary_writer, save_on_best):
    """External evaluation such as BLEU and ROUGE scores."""
    out_dir = hparams.out_dir
    decode = global_step > 0
    
    if decode:
        utils.print_out("# External evaluation, global step %d" % global_step)

    # TF2: Set variables instead of feed_dict
    for key, value in iterator_feed_dict.items():
        key.assign(value)

    output = os.path.join(out_dir, "output_%s" % label)
    scores = nmt_utils.decode_and_evaluate(
        label, 
        model, 
        output, 
        ref_file=tgt_file, 
        metrics=hparams.metrics, 
        bpe_delimiter=hparams.bpe_delimiter, 
        beam_width=hparams.beam_width, 
        tgt_eos=hparams.eos, 
        decode=decode
    )

    # Save on best metrics
    if decode:
        for metric in hparams.metrics:
            utils.add_summary(
                summary_writer, global_step, 
                "%s_%s" % (label, metric), scores[metric]
            )
            # metric: larger is better
            if save_on_best and scores[metric] > getattr(hparams, "best_" + metric):
                setattr(hparams, "best_" + metric, scores[metric])
                model.save_weights(
                    os.path.join(
                        getattr(hparams, "best_" + metric + "_dir"), 
                        "translate.ckpt"
                    )
                )
        utils.save_hparams(out_dir, hparams)
    
    return scores