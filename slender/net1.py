import abc
import os
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1

from .blob import Blob
from .util import scope_join_fn

slim = tf.contrib.slim

_ = scope_join_fn('net')

_NET = resnet_v1.resnet_v1_50
_SIZE = resnet_v1.resnet_v1.default_image_size
_ARG_SCOPE_FN = resnet_v1.resnet_arg_scope

# TODO: eliminate this section
_NET_SCOPE = _NET.__name__
__ = scope_join_fn(_NET_SCOPE)
_CKPT_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)),
    os.pardir,
    'model',
    _NET_SCOPE + '.ckpt',
)


class BaseNet(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_scope_set(scopes=[''], collection=tf.GraphKeys.VARIABLES):
        return set().union(*[slim.get_variables(scope, collection=collection) for scope in scopes])

    def __init__(self,
                 working_dir,
                 num_classes,
                 is_training,
                 weight_decay,
                 gpu_frac,
                 log_device_placement,
                 verbosity):

        self.working_dir = working_dir
        self.num_classes = num_classes
        self.arg_scope = _ARG_SCOPE_FN(
            is_training=is_training,
            weight_decay=weight_decay,
        )
        self.session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac),
            log_device_placement=log_device_placement,
        )
        self.graph = tf.get_default_graph()

        tf.logging.set_verbosity(verbosity)

    def forward(self):
        with slim.arg_scope(self.arg_scope):
            (self.feat_maps, self.end_points) = _NET(
                self.images,
                global_pool=False,
                scope=_NET_SCOPE,
            )

        return Blob(
            feat_maps=self.feat_maps,
            labels=self.labels,
        )

    @abc.abstractmethod
    def eval(self, blob):
        pass

    @abc.abstractmethod
    def run(self):
        pass


class TrainNet(BaseNet):
    _SCOPES_TO_FREEZE = [
        __('conv1'),
        __('block1'),
        __('block2'),
    ]

    def __init__(self,
                 working_dir,
                 num_classes,
                 scopes_to_freeze=_SCOPES_TO_FREEZE,
                 metric_attrs,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 weight_decay=1e-3,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super

        self.scopes_to_freeze = scopes_to_freeze
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate

    def eval(self):
        self.global_step = slim.get_or_create_global_step()
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

        all_vars = BaseNet.get_scope_set()
        self.vars_to_restore = BaseNet.get_scope_set(_NET_SCOPE)
        self.vars_to_train = (
            BaseNet.get_scope_set(collection=tf.GraphKeys.TRAINABLE_VARIABLES) -
            BaseNet.get_scope_set(self.scopes_to_freeze)
        )
        self.vars_to_save = all_vars

        self.metrics = {
            'train/{:s}'.format(attr): self.__getattribute__(attr)
            for attr in self.metric_attrs
        }
        self.summary_ops = map(tf.scalar_summary, self.metrics.keys(), self.metrics.values())

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1.0)
        self.train_op = slim.learning.create_train_op(
            self.total_loss,
            self.optimizer,
            variables_to_train=self.vars_to_train,
        )
        (assign_op, assign_feed_dict) = slim.assign_from_checkpoint(_CKPT_PATH, list(self.vars_to_restore))
        init_op = tf.initialize_variables(all_vars - self.vars_to_restore)
        self.init_op = tf.group(assign_op, init_op)
        self.init_feed_dict = assign_feed_dict
        self.saver = tf.train.Saver(self.vars_to_save)

    def run(self,
            number_of_steps,
            log_every_n_steps=1,
            save_summaries_secs=10,
            save_interval_secs=600):

        slim.learning.train(
            self.train_op,
            self.working_dir,
            log_every_n_steps=log_every_n_steps,
            number_of_steps=number_of_steps,
            init_op=self.init_op,
            init_feed_dict=self.init_feed_dict,
            save_summaries_secs=save_summaries_secs,
            saver=self.saver,
            save_interval_secs=save_interval_secs,
            session_config=self.session_config,
        )


class ClassifyNet(BaseNet):
    def forward(self, blob):
        super(ClassifyNet, self).forward()

        self.forward_var_scope = _('forward')
        with slim.arg_scope(self.arg_scope), tf.variable_scope(self.forward_var_scope):
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
            self.predictions = slim.softmax(
                self.logits,
                scope='predictions',
            )

            self.predictions_flattened = tf.squeeze(self.predictions, (1, 2))
            self.targets = tf.one_hot(self.labels, self.num_classes)
            self.loss = slim.losses.log_loss(self.predictions_flattened, self.targets, weight=self.num_classes)
            self.total_loss = slim.losses.get_total_loss()

            self.predicted_labels = tf.argmax(self.predictions_flattened, 1)
            self.accuracy = slim.metrics.accuracy(self.predicted_labels, self.labels)

