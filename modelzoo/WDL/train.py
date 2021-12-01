# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# This script used to create and train wide and deep model on Kaggle's Criteo Dataset

import time
import argparse
import tensorflow as tf
import math
import collections
import numpy as np
import os
import sys

from tensorflow.python.ops import partitioned_variables

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))
CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
IDENTIY_COLUMNS = ["I10"]
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 300000,
    'C4': 250000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 250000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 200000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 250000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}

IDENTITY_NUM_BUCKETS = {'I10': 10}

EMBEDDING_DIMENSIONS = {
    'C1': 64,
    'C2': 64,
    'C3': 128,
    'C4': 128,
    'C5': 64,
    'C6': 64,
    'C7': 64,
    'C8': 64,
    'C9': 64,
    'C10': 128,
    'C11': 64,
    'C12': 128,
    'C13': 64,
    'C14': 64,
    'C15': 64,
    'C16': 128,
    'C17': 64,
    'C18': 64,
    'C19': 64,
    'C20': 64,
    'C21': 128,
    'C22': 64,
    'C23': 64,
    'C24': 128,
    'C25': 64,
    'C26': 128
}


def generate_input_fn(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        cont_defaults = [[0.0] for i in range(1, 10)]
        cont_defaults = cont_defaults + [[0]]
        cont_defaults = cont_defaults + [[0.0] for i in range(11, 14)]
        cate_defaults = [[" "] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=2021)  # fix seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(1)
    return dataset


def build_feature_cols(train_file_path, test_file_path):
    # Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time
    print('****Computing statistics of train dataset*****')
    # with open(train_file_path, 'r') as f, open(test_file_path, 'r') as f1:
    #     nums = [line.strip('\n').split(',') for line in f.readlines()
    #             ] + [line.strip('\n').split(',') for line in f1.readlines()]
    #     numpy_arr = np.array(nums)
    #     mins_list, max_list, range_list = [], [], []
    #     for i in range(len(TRAIN_DATA_COLUMNS)):
    #         if TRAIN_DATA_COLUMNS[i] in CONTINUOUS_COLUMNS:
    #             col_min = numpy_arr[:, i].astype(np.float32).min()
    #             col_max = numpy_arr[:, i].astype(np.float32).max()
    #             mins_list.append(col_min)
    #             max_list.append(col_max)
    #             range_list.append(col_max - col_min)
    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):
        def minmaxscaler(col):
            return (col - min) / range

        return minmaxscaler

    deep_columns = []
    wide_columns = []
    embedding_columns = []
    for column_name in FEATURE_COLUMNS:
        if column_name in IDENTITY_NUM_BUCKETS:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=IDENTITY_NUM_BUCKETS[column_name])
            wide_columns.append(categorical_column)
            deep_columns.append(
                tf.feature_column.indicator_column(categorical_column))
        elif column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                dtype=tf.string)
            wide_columns.append(categorical_column)

            embedding_columns.append(
                tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    combiner='mean'))
        else:
            normalizer_fn = None
            i = CONTINUOUS_COLUMNS.index(column_name)
            col_min = mins_list[i]
            col_range = range_list[i]
            normalizer_fn = make_minmaxscaler(col_min, col_range)
            column = tf.feature_column.numeric_column(
                column_name, normalizer_fn=normalizer_fn, shape=(1, ))
            wide_columns.append(column)
            deep_columns.append(column)

    return wide_columns, deep_columns, embedding_columns


def wide_optimizer():
    opt = tf.train.FtrlOptimizer(
        learning_rate=args.linear_learning_rate,
        l1_regularization_strength=args.linear_l1_regularization,
        l2_regularization_strength=args.linear_l2_regularization)
    return opt


def deep_optimizer():
    learning_rate_fn = args.deep_learning_rate
    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate_fn,
                                    initial_accumulator_value=0.1,
                                    use_locking=False)
    return opt


def build_estimator(model_dir=None,
                    train_file_path=None,
                    test_file_path=None,
                    wide_optimizer=None,
                    deep_optimizer=None,
                    deep_dropout=0.0,
                    deep_hidden_units=[1024, 512, 256]):
    if model_dir is None:
        model_dir = 'models/model_WIDE_AND_DEEP_' + str(int(time.time()))
        print("Model directory = %s" % model_dir)

    wide_columns, deep_columns, embedding_columns = build_feature_cols(
        train_file_path, test_file_path)
    deep_columns += embedding_columns

    sess_config = tf.ConfigProto()
    if args.inter:
        sess_config.inter_op_parallelism_threads = args.inter
    if args.intra:
        sess_config.intra_op_parallelism_threads = args.intra

    runconfig = tf.estimator.RunConfig(
        save_checkpoints_steps=args.save_steps,
        keep_checkpoint_max=1,
        tf_random_seed=2021,
        protocol=args.protocol,
        session_config=sess_config)  # fix random seed for reproducing

    num_ps_replicas = runconfig.num_ps_replicas
    # if num_ps_replicas:
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.input_layer_partitioner <<
        20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.dense_layer_partitioner <<
        10) if args.dense_layer_partitioner else None

    if args.bf16:
        m = tf.estimator.DNNLinearCombinedClassifier(
            config=runconfig,
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=wide_optimizer,
            linear_sparse_combiner='sum',
            dnn_feature_columns=deep_columns,
            dnn_optimizer=deep_optimizer,
            dnn_dropout=deep_dropout,
            dnn_dtype=tf.bfloat16,
            dnn_hidden_units=deep_hidden_units,
            input_layer_partitioner=input_layer_partitioner,
            dense_layer_partitioner=dense_layer_partitioner,
            loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    else:
        if dense_layer_partitioner:
            m = tf.estimator.DNNLinearCombinedClassifier(
                config=runconfig,
                model_dir=model_dir,
                linear_feature_columns=wide_columns,
                linear_optimizer=wide_optimizer,
                linear_sparse_combiner='sum',
                dnn_feature_columns=deep_columns,
                dnn_optimizer=deep_optimizer,
                dnn_dropout=deep_dropout,
                dnn_hidden_units=deep_hidden_units,
                input_layer_partitioner=input_layer_partitioner,
                dense_layer_partitioner=dense_layer_partitioner,
                loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        else:
            m = tf.estimator.DNNLinearCombinedClassifier(
                config=runconfig,
                model_dir=model_dir,
                linear_feature_columns=wide_columns,
                linear_optimizer=wide_optimizer,
                linear_sparse_combiner='sum',
                dnn_feature_columns=deep_columns,
                dnn_optimizer=deep_optimizer,
                dnn_dropout=deep_dropout,
                dnn_hidden_units=deep_hidden_units,
                input_layer_partitioner=input_layer_partitioner,
                loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    print('estimator built')
    return m


# All categorical columns are strings for this dataset
def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--deep_dropout',
                        help='Dropout regularization for deep model',
                        type=float,
                        default=0.0)
    parser.add_argument('--linear_l1_regularization',
                        help='L1 regularization for linear model',
                        type=float,
                        default=0.0)
    parser.add_argument('--linear_l2_regularization',
                        help='L2 regularization for linear model',
                        type=float,
                        default=0.0)
    parser.add_argument('--linear_learning_rate',
                        help='Learning rate for linear model',
                        type=float,
                        default=0.2)
    parser.add_argument('--deep_learning_rate',
                        help='Learning rate for deep model',
                        type=float,
                        default=0.05)
    parser.add_argument('--timeline',
                        help='Steps to save timeline, zero to close',
                        type=int,
                        default=0)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument("--protocol",
                        type=str,
                        choices=["grpc", "grpc++", "star_server"],
                        default="grpc")
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=0)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=0)
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    print("Begin training and evaluation")
    # check dataset
    train_file = args.data_location + '/train.csv'
    test_file = args.data_location + '/eval.csv'
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print(
            '------------------------------------------------------------------------------------------'
        )
        print(
            "train.csv or eval.csv does not exist in the given data_location. Please provide valid path"
        )
        print(
            '------------------------------------------------------------------------------------------'
        )
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set batch size & steps
    batch_size = args.batch_size
    if args.steps == 0:
        no_of_epochs = 10
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_WIDE_AND_DEEP_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)
    export_dir = args.output_dir

    # create model by estimator
    m = build_estimator(checkpoint_dir, train_file, test_file, wide_optimizer,
                        deep_optimizer, args.deep_dropout, [1024, 512, 256])

    # add hooks
    hooks = []
    if args.timeline > 0:
        hooks.append(
            tf.estimator.ProfilerHook(save_steps=args.timeline,
                                      output_dir=checkpoint_dir))

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        # stand-alone training
        # train model
        m.train(input_fn=lambda: generate_input_fn(train_file, batch_size,
                                                   int(no_of_epochs)),
                steps=int(train_steps),
                hooks=hooks)
        print('fit done')

        # evaluate model
        if not args.no_eval:
            results = m.evaluate(
                input_fn=lambda: generate_input_fn(test_file, batch_size, 1),
                steps=test_steps)
            print('evaluate done')

            print('Accuracy: %s' % results['accuracy'])
            print('AUC: %s' % results['auc'])
    else:
        # distribute training
        # train & evaluate model
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: generate_input_fn(
            train_file, batch_size, int(no_of_epochs)),
                                            max_steps=int(train_steps),
                                            hooks=hooks)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: generate_input_fn(test_file, batch_size, 1),
            steps=test_steps)
        results = tf.estimator.train_and_evaluate(m, train_spec, eval_spec)
        print(results)