import abc
import os
import tensorflow as tf

from .blob import Blob
from .util import scope_join_fn

slim = tf.contrib.slim


class BaseNet(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_scope_set(scopes=None, collection=tf.GraphKeys.VARIABLES):
        scopes = scopes or ['']
        return set().union(*[
            slim.get_variables(scope, collection=collection)
            for scope in scopes
        ])

    def __init__(self,
                 num_classes,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        self.num_classes = num_classes
        self.session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac),
            log_device_placement=log_device_placement,
        )
        self.graph = tf.get_default_graph()

        tf.logging.set_verbosity(verbosity)

    @abc.abstractmethod
    def forward(self, blob):
        pass


class ResNet50(BaseNet):
    def __init__(self,
                 is_training,
                 num_classes,
                 weight_decay=1e-3,
                 ckpt_path=None,
                 scopes_to_restore=None,
                 scopes_to_freeze=None,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        from tensorflow.contrib.slim.nets import resnet_v1

        self.__net = resnet_v1
        self.__var_scope = 'resnet_v1_50'
        self.__scope_join = scope_join_fn(self.__var_scope)
        self.__ckpt_path = os.path.join(
            os.path.realpath(os.path.dirname(__file__)),
            os.pardir,
            'model',
            'resnet_v1_50.ckpt',
        )

        self.__scopes_to_restore = [
            self.__var_scope,
        ]
        self.__scopes_to_freeze = [
            self.__scope_join(scope) for scope in [
                'conv1',
                'block1',
                'block2',
            ]
        ]

        self.ckpt_path = ckpt_path or self.__ckpt_path
        self.scopes_to_restore = scopes_to_restore or self.__scopes_to_restore
        self.scopes_to_freeze = scopes_to_freeze or self.__scopes_to_freeze
        self.arg_scope = self.__net.resnet_arg_scope(
            is_training=is_training,
            weight_decay=weight_decay,
        )

        super(ResNet50, self).__init__(
            num_classes=num_classes,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

    def forward(self, blob):
        with slim.arg_scope(self.arg_scope):
            (feat_maps, _) = self.__net.resnet_v1_50(
                blob.images,
                global_pool=False,
                scope=self.__var_scope,
            )

        return Blob(
            feat_maps=feat_maps,
            labels=blob.labels,
        )


class BaseMixin(object):
    def __init__(self,
                 scope,
                 working_dir):

        self.__scope = scope
        self.__scope_join = scope_join_fn(self.__scope)
        self.working_dir = working_dir

    def eval(self, blob):
        summary_ops = [
            tf.scalar_summary(self.__scope_join(key), value)
            for (key, value) in blob.items()
        ]


class TrainMixin(BaseMixin):
    def __init__(self,
                 working_dir,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5):

        super(TrainMixin, self).__init__(
            scope='train',
            working_dir=working_dir,
        )

        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate

    def eval(self, blob):
        super(TrainMixin, self).eval(blob)

        vars_to_restore = BaseNet.get_scope_set(self.scopes_to_restore)
        vars_to_train = (
            BaseNet.get_scope_set(collection=tf.GraphKeys.TRAINABLE_VARIABLES) -
            BaseNet.get_scope_set(self.scopes_to_freeze)
        )

        learning_rate = tf.constant(
            self.learning_rate,
            dtype=tf.float32,
            name='learning_rate',
        )
        if self.learning_rate_decay_steps is not None:
            global_step = slim.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                learning_rate,
                global_step=global_step,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay_rate,
                staircase=True,
                name='decaying_learning_rate',
            )

        total_loss = slim.losses.get_total_loss()
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
        self.train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            variables_to_train=vars_to_train,
        )

        all_vars = BaseNet.get_scope_set()
        init_op = tf.initialize_variables(all_vars - vars_to_restore)
        (assign_op, assign_feed_dict) = slim.assign_from_checkpoint(
            self.ckpt_path,
            list(vars_to_restore),
        )

        self.init_op = tf.group(assign_op, init_op)
        self.init_feed_dict = assign_feed_dict
        self.saver = tf.train.Saver(all_vars)

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


class TestMixin(object):
    def __init__(self,
                 working_dir):

        super(TestMixin, self).__init__(
            scope='test',
            working_dir=working_dir,
        )

    def eval(self, blob):
        (values, _) = slim.metrics.aggregate_metrics(*[
            slim.metrics.streaming_mean(value)
            for value in blob.values()
        ])

        blob = Blob({
            key: value
            for (key, value) in zip(blob.keys(), values)
        })

        super(TestMixin, self).eval(blob)

    def run(self,
            num_steps,
            eval_interval_secs=300,
            timeout=600):

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


class ClassifyNet(ResNet50):
    def __init__(self,
                 is_training,
                 num_classes,
                 weight_decay=1e-3,
                 scopes_to_restore=None,
                 scopes_to_freeze=None,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(ClassifyNet, self).__init__(
            is_training=is_training,
            num_classes=num_classes,
            weight_decay=weight_decay,
            scopes_to_restore=scopes_to_restore,
            scopes_to_freeze=scopes_to_freeze,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        self.__var_scope = 'classify_net'
        self.__scope_join = scope_join_fn(self.__var_scope)

    def forward(self, blob):
        blob = super(ClassifyNet, self).forward(blob)

        with slim.arg_scope(self.arg_scope), tf.variable_scope(self.__var_scope):
            feats = tf.reduce_mean(
                blob.feat_maps,
                (1, 2),
                keep_dims=True,
                name='feats',
            )
            logits = slim.conv2d(
                feats,
                self.num_classes,
                (1, 1),
                activation_fn=None,
                normalizer_fn=None,
                scope='logits',
            )
            predictions = slim.softmax(
                logits,
                scope='predictions',
            )

            predictions = tf.squeeze(predictions, (1, 2))
            targets = tf.one_hot(blob.labels, self.num_classes)
            loss = slim.losses.log_loss(predictions, targets, weight=self.num_classes)

            predicted_labels = tf.argmax(predictions, 1)
            accuracy = slim.metrics.accuracy(predicted_labels, blob.labels)

        return Blob(
            loss=loss,
            accuracy=accuracy,
        )


class TrainClassifyNet(ClassifyNet, TrainMixin):
    def __init__(self,
                 working_dir,
                 num_classes,
                 weight_decay=1e-3,
                 scopes_to_restore=None,
                 scopes_to_freeze=None,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        ClassifyNet.__init__(
            self,
            is_training=True,
            num_classes=num_classes,
            weight_decay=weight_decay,
            scopes_to_restore=scopes_to_restore,
            scopes_to_freeze=scopes_to_freeze,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        TrainMixin.__init__(
            self,
            working_dir=working_dir,
            learning_rate=learning_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
        )


class TestClassifyNet(ClassifyNet, TestMixin):
    def __init__(self,
                 working_dir,
                 num_classes,
                 weight_decay=1e-3,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        ClassifyNet.__init__(
            self,
            is_training=False,
            num_classes=num_classes,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        TestMixin.__init__(
            self,
            working_dir=working_dir,
        )
