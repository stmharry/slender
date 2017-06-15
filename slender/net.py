# -*- encoding: utf-8 -*-

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .blob import Blob
from .util import scope_join_fn


class BaseNet(object):
    IS_TRAINING = None  # set by scheme

    @staticmethod
    def get_scope_set(scopes=None, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        scopes = scopes or ['']
        return set().union(*[
            slim.get_variables(scope, collection=collection)
            for scope in scopes
        ])

    def __init__(self,
                 working_dir,
                 ckpt_path,
                 scopes_to_restore,
                 scopes_to_freeze,
                 summary_attrs,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        self.working_dir = working_dir
        self.ckpt_path = ckpt_path
        self.scopes_to_restore = scopes_to_restore
        self.scopes_to_freeze = scopes_to_freeze
        self.summary_attrs = summary_attrs
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate

        self.global_step = tf.train.get_or_create_global_step()
        self.session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac),
            log_device_placement=log_device_placement,
        )
        self.graph = tf.get_default_graph()

        tf.logging.set_verbosity(verbosity)

    def build(self, blob):
        blob = self.forward(blob)
        self.prepare()
        return blob

    def forward(self, blob):  # implemented by net
        pass

    def prepare(self):  # implemented by scheme
        pass

    def run(self, **kwargs):
        pass


class ResNet50(BaseNet):
    VAR_SCOPE = 'resnet_v1_50'
    CKPT_PATH = os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        os.pardir,
        'model',
        'resnet_v1_50.ckpt',
    )
    SCOPES_TO_RESTORE = [
        None,
    ]
    SCOPES_TO_FREEZE = [
        'conv1',
        'block1',
        'block2',
    ]
    SUMMARY_ATTRS = []

    def __init__(self,
                 working_dir=None,
                 weight_decay=1e-3,
                 ckpt_path=None,
                 scope=None,
                 scopes_to_restore=None,
                 scopes_to_freeze=None,
                 summary_attrs=None,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        from tensorflow.contrib.slim.nets import resnet_v1
        scope = None  # TODO: allow remapping when assigning from ckpt

        self.__net = resnet_v1
        self.__var_scope = scope_join_fn(scope)(ResNet50.VAR_SCOPE)
        self.__scope_join = scope_join_fn(self.__var_scope)

        _scopes_to_restore = map(self.__scope_join, ResNet50.SCOPES_TO_RESTORE)
        _scopes_to_freeze = map(self.__scope_join, ResNet50.SCOPES_TO_FREEZE)

        self.arg_scope = self.__net.resnet_arg_scope(
            is_training=self.IS_TRAINING,
            weight_decay=weight_decay,
        )

        super(ResNet50, self).__init__(
            working_dir,
            ckpt_path=ckpt_path or ResNet50.CKPT_PATH,
            scopes_to_restore=scopes_to_restore or _scopes_to_restore,
            scopes_to_freeze=scopes_to_freeze or _scopes_to_freeze,
            summary_attrs=summary_attrs or ResNet50.SUMMARY_ATTRS,
            learning_rate=learning_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

    def forward(self, blob):
        with slim.arg_scope(self.arg_scope):
            (feat_maps, _) = self.__net.resnet_v1_50(
                blob['images'],
                global_pool=False,
                scope=self.__var_scope,
            )

        return Blob(
            feat_maps=feat_maps,
            labels=blob['labels'],
        )


class BaseScheme(BaseNet):
    VAR_SCOPE = None

    @classmethod
    def get_working_dir(cls, working_dir):
        return os.path.join(working_dir, cls.VAR_SCOPE)


class TrainScheme(BaseScheme):
    VAR_SCOPE = 'train'
    IS_TRAINING = True

    def prepare(self):
        self.summary_ops = [
            tf.summary.scalar(
                scope_join_fn(TrainScheme.VAR_SCOPE)(attr),
                self.__getattribute__(attr),
            )
            for attr in self.summary_attrs
        ]

        with tf.variable_scope(TrainScheme.VAR_SCOPE):
            all_model_vars = BaseNet.get_scope_set()
            vars_to_restore = BaseNet.get_scope_set(self.scopes_to_restore)
            vars_to_train = (
                BaseNet.get_scope_set(collection=tf.GraphKeys.TRAINABLE_VARIABLES) -
                BaseNet.get_scope_set(self.scopes_to_freeze)
            )

            self.learning_rate = tf.constant(
                self.learning_rate,
                dtype=tf.float32,
                name='learning_rate',
            )
            if self.learning_rate_decay_steps is not None:
                self.learning_rate = tf.train.exponential_decay(
                    self.learning_rate,
                    global_step=self.global_step,
                    decay_steps=self.learning_rate_decay_steps,
                    decay_rate=self.learning_rate_decay_rate,
                    staircase=True,
                    name='decaying_learning_rate',
                )

            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1.0)
            self.train_op = slim.learning.create_train_op(
                self.total_loss,
                optimizer,
                variables_to_train=list(vars_to_train),
            )

            all_vars = BaseNet.get_scope_set()
            init_op = tf.variables_initializer(all_vars - vars_to_restore)
            (assign_op, assign_feed_dict) = slim.assign_from_checkpoint(
                self.ckpt_path,
                list(vars_to_restore),
            )

            self.init_op = tf.group(assign_op, init_op)
            self.init_feed_dict = assign_feed_dict
            self.saver = tf.train.Saver(all_model_vars)

    def run(self,
            number_of_steps,
            log_every_n_steps=1,
            save_summaries_secs=10,
            save_interval_secs=600):

        slim.learning.train(
            train_op=self.train_op,
            logdir=TrainScheme.get_working_dir(self.working_dir),
            log_every_n_steps=log_every_n_steps,
            number_of_steps=number_of_steps,
            init_op=self.init_op,
            init_feed_dict=self.init_feed_dict,
            save_summaries_secs=save_summaries_secs,
            saver=self.saver,
            save_interval_secs=save_interval_secs,
            session_config=self.session_config,
        )


class TestScheme(BaseScheme):
    VAR_SCOPE = 'test'
    IS_TRAINING = False

    def prepare(self):
        with tf.variable_scope(TestScheme.VAR_SCOPE):
            (values, update_ops) = slim.metrics.aggregate_metrics(*[
                slim.metrics.streaming_mean(self.__getattribute__(attr))
                for attr in self.summary_attrs
            ])
            self.eval_op = tf.group(*update_ops)

            for (attr, value) in zip(self.summary_attrs, values):
                self.__setattr__(attr, value)

        self.summary_ops = [
            tf.summary.scalar(
                scope_join_fn(TestScheme.VAR_SCOPE)(attr),
                self.__getattribute__(attr),
            )
            for attr in self.summary_attrs
        ]
        self.all_model_vars = BaseNet.get_scope_set()

    def run(self,
            num_steps,
            eval_interval_secs=300,
            timeout=600):

        slim.evaluation.evaluation_loop(
            master='',
            checkpoint_dir=TrainScheme.get_working_dir(self.working_dir),
            logdir=TestScheme.get_working_dir(self.working_dir),
            num_evals=num_steps,
            eval_op=self.eval_op,
            eval_interval_secs=eval_interval_secs,
            variables_to_restore=self.all_model_vars,
            session_config=self.session_config,
            timeout=timeout,
        )


class OnlineScheme(BaseScheme):
    VAR_SCOPE = 'online'
    IS_TRAINING = False

    def prepare(self):
        with tf.variable_scope(OnlineScheme.VAR_SCOPE):
            vars_to_restore = BaseNet.get_scope_set()

            (assign_op, assign_feed_dict) = slim.assign_from_checkpoint(
                tf.train.latest_checkpoint(TrainScheme.get_working_dir(self.working_dir)),
                list(vars_to_restore),
            )

            self.init_op = assign_op
            self.init_feed_dict = assign_feed_dict

    def run(self):
        self.sess = tf.Session(
            graph=self.graph,
            config=self.session_config,
        )
        self.sess.run(self.init_op, feed_dict=self.init_feed_dict)
        tf.train.start_queue_runners(sess=self.sess)


class ClassifyNet(ResNet50):
    VAR_SCOPE = 'classify_net'
    SUMMARY_ATTRS = ['loss', 'total_loss', 'accuracy']
    OUTPUT_ATTRS = ['predictions', 'labels']

    def __init__(self,
                 num_classes,
                 working_dir=None,
                 weight_decay=1e-3,
                 ckpt_path=None,
                 scope=None,
                 scopes_to_restore=None,
                 scopes_to_freeze=None,
                 summary_attrs=None,
                 output_attrs=None,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        self.__var_scope = scope_join_fn(scope)(ClassifyNet.VAR_SCOPE)
        self.__scope_join = scope_join_fn(self.__var_scope)

        super(ClassifyNet, self).__init__(
            working_dir=working_dir,
            weight_decay=weight_decay,
            ckpt_path=ckpt_path,
            scope=self.__var_scope,
            scopes_to_restore=scopes_to_restore,
            scopes_to_freeze=scopes_to_freeze,
            summary_attrs=summary_attrs or ClassifyNet.SUMMARY_ATTRS,
            learning_rate=learning_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        self.num_classes = num_classes
        self.output_attrs = output_attrs or ClassifyNet.OUTPUT_ATTRS

    def forward(self, blob):
        blob = super(ClassifyNet, self).forward(blob)
        self.feat_maps = blob['feat_maps']
        self.labels = blob['labels']

        with slim.arg_scope(self.arg_scope), tf.variable_scope(self.__var_scope):
            self.feats = tf.reduce_mean(
                self.feat_maps,
                (1, 2),
                keep_dims=True,
                name='feats',
            )
            self.logits = slim.conv2d(
                self.feats,
                self.num_classes,
                (1, 1),
                activation_fn=None,
                normalizer_fn=None,
                scope='logits',
            )
            self.logits = tf.squeeze(
                self.logits,
                axis=(1, 2),
            )
            self.predictions = tf.nn.softmax(
                self.logits,
                name='predictions',
            )

            # '''
            self.loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.labels,
                logits=self.logits,
                scope='loss',
            )
            '''
            self.targets = tf.one_hot(self.labels, self.num_classes)
            self.loss = tf.losses.log_loss(
                labels=self.targets,
                predictions=self.predictions,
                weights=self.num_classes,
            )
            '''
            self.total_loss = tf.losses.get_total_loss()

            self.predicted_labels = tf.argmax(self.predictions, 1)
            self.accuracy = slim.metrics.accuracy(
                labels=self.labels,
                predictions=self.predicted_labels,
            )

        return Blob(**{
            attr: self.__getattribute__(attr)
            for attr in self.output_attrs
        })
