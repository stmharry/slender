import abc
import os
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1

from .blob import Blob
from .util import scope_join

slim = tf.contrib.slim

_NET = resnet_v1.resnet_v1_50
_SIZE = resnet_v1.resnet_v1.default_image_size
_ARG_SCOPE_FN = resnet_v1.resnet_arg_scope

_SCOPE = _NET.__name__
_CKPT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model', _SCOPE + '.ckpt')


def _(name):
    return scope_join(_SCOPE, name)


class BaseNet(object):
    __metaclass__ = abc.ABCMeta

    _SCOPES_TO_RESTORE = [
        _(''),
    ]
    _SCOPES_NOT_TO_RESTORE = [
        _('logits'),
    ]
    _SCOPES_TO_TRAIN = [
        _('block3'),
        _('block4'),
        _('logits'),
    ]

    _METRIC_ATTRS = [
        'loss',
        'total_loss',
        'entropy_factor',
        'accuracy',
    ]
    _OUTPUT_ATTRS = [
        'logits',
        'predictions',
    ]

    @staticmethod
    def get_scopes(scopes=[''], collection=tf.GraphKeys.VARIABLES):
        return set().union(*[slim.get_variables(scope, collection=collection) for scope in scopes])

    def __init__(self,
                 working_dir,
                 num_classes,
                 is_training,
                 gpu_frac,
                 log_device_placement,
                 verbosity):

        self.working_dir = working_dir
        self.num_classes = num_classes
        self.arg_scope = _ARG_SCOPE_FN(is_training=is_training)
        self.session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac),
            log_device_placement=log_device_placement,
        )

        tf.logging.set_verbosity(verbosity)

    @abc.abstractmethod
    def _forward(self):
        pass

    def forward(self, blob):
        self.images = blob.image
        self.labels = blob.label

        with slim.arg_scope(self.arg_scope):
            (net, self.end_points) = _NET(
                self.images,
                num_classes=self.num_classes,
                global_pool=True,
                scope=_SCOPE,
            )

        with tf.variable_scope(_('forward')):
            self.logits = tf.squeeze(self.end_points[_('logits')], (1, 2))
            self.predictions = tf.squeeze(self.end_points['predictions'], (1, 2))

            self.targets = tf.one_hot(self.labels, self.num_classes)
            self.loss = slim.losses.log_loss(self.predictions, self.targets)
            self.entropy_factor = tf.exp(self.loss * self.num_classes)
            self.total_loss = slim.losses.get_total_loss()

            self.predicted_labels = tf.argmax(self.predictions, 1)
            self.accuracy = slim.metrics.accuracy(self.predicted_labels, self.labels)

            self.metric_names = BaseNet._METRIC_ATTRS
            self.metric_values = map(self.__getattribute__, BaseNet._METRIC_ATTRS)

        self._forward()

        return Blob(**dict(zip(
            BaseNet._OUTPUT_ATTRS,
            map(self.__getattribute__, BaseNet._OUTPUT_ATTRS),
        )))


class TrainNet(BaseNet):
    def __init__(self,
                 working_dir,
                 num_classes,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(TrainNet, self).__init__(
            working_dir=working_dir,
            num_classes=num_classes,
            is_training=True,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        self.global_step = slim.get_or_create_global_step()
        self.learning_rate = tf.constant(
            learning_rate,
            dtype=tf.float32,
            name='learning_rate',
        )
        if learning_rate_decay_steps is not None:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=learning_rate_decay_steps,
                decay_rate=learning_rate_decay_rate,
                staircase=True,
                name='decaying_learning_rate',
            )

    def _forward(self):
        self.variables_to_restore = list(
            BaseNet.get_scopes(BaseNet._SCOPES_TO_RESTORE) -
            BaseNet.get_scopes(BaseNet._SCOPES_NOT_TO_RESTORE)
        )
        self.variables_to_save = list(
            BaseNet.get_scopes()
        )
        self.variables_to_train = list(
            BaseNet.get_scopes(BaseNet._SCOPES_TO_TRAIN) &
            BaseNet.get_scopes(collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        )

        self.metric_names = map('train/'.__add__, self.metric_names)
        self.summary_ops = map(tf.scalar_summary, self.metric_names, self.metric_values)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1.0)
        self.train_op = slim.learning.create_train_op(
            self.total_loss,
            self.optimizer,
            variables_to_train=self.variables_to_train,
        )

    def train(self,
              number_of_steps,
              log_every_n_steps=1,
              save_summaries_secs=10,
              save_interval_secs=600):

        slim.learning.train(
            self.train_op,
            self.working_dir,
            log_every_n_steps=log_every_n_steps,
            number_of_steps=number_of_steps,
            init_fn=slim.assign_from_checkpoint_fn(_CKPT_PATH, self.variables_to_restore),
            save_summaries_secs=save_summaries_secs,
            saver=tf.train.Saver(self.variables_to_save),
            save_interval_secs=save_interval_secs,
            session_config=self.session_config,
        )


class TestNet(BaseNet):
    def __init__(self,
                 working_dir,
                 num_classes,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(TestNet, self).__init__(
            working_dir=working_dir,
            num_classes=num_classes,
            gpu_frac=gpu_frac,
            is_training=False,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

    def _forward(self):
        self.metric_names = map('test/'.__add__, self.metric_names)
        self.metric_value_updates = map(slim.metrics.streaming_mean, self.metric_values)
        (self.metric_values, self.metric_updates) = slim.metrics.aggregate_metrics(*self.metric_value_updates)
        self.summary_ops = map(tf.scalar_summary, self.metric_names, self.metric_values)

    def test(self,
             num_steps,
             eval_interval_secs=60,
             timeout=900):

        slim.evaluation.evaluation_loop(
            '',
            self.working_dir,
            os.path.join(self.working_dir, 'test'),
            num_evals=num_steps,
            eval_op=self.metric_updates,
            eval_interval_secs=eval_interval_secs,
            session_config=self.session_config,
            timeout=timeout,
        )


class OnlineNet(BaseNet):
    def __init__(self,
                 working_dir,
                 num_classes,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(OnlineNet, self).__init__(
            working_dir=working_dir,
            num_classes=num_classes,
            gpu_frac=gpu_frac,
            is_training=False,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        self.sess = None

    def _forward(self):
        self.variables_to_restore = list(BaseNet.get_scopes())
        (self.init_assign_op, self.init_feed_dict) = slim.assign_from_checkpoint(tf.train.latest_checkpoint(self.working_dir), self.variables_to_restore)

    def init(self, blob):
        self.sess = tf.Session(
            config=self.session_config,
        )
        self.sess.run(self.init_assign_op, feed_dict=self.init_feed_dict)

        return Blob(sess=self.sess, **blob._dict)

    def online(self, blob, feed_dict=None):
        blob_val = blob.eval(self.sess, feed_dict=feed_dict)
        return blob_val
