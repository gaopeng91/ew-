import tensorflow as tf
from tensorcv.net import get_net_fn
from tensorcv.train.loss import get_loss_fn
from tensorcv.train.predictions import get_predictions_fn
from tensorcv.train.metrics import get_metrics_fn
from tensorcv.train.lr_policy import get_lr_policy_fn
from tensorcv.train.optimizer import get_optimizer_fn
from tensorcv.train.summary import get_summary_fn


class Model(object):
    def __init__(self, config):
        self.config = config
        self._net_fn = get_net_fn(config)
        self._loss_fn = get_loss_fn(config)
        self._predictions_fn = get_predictions_fn(config)
        self._metrics_fn = get_metrics_fn(config)
        self._lr_policy_fn = get_lr_policy_fn(config)
        self._optimizer_fn = get_optimizer_fn(config)
        self._summary_fn = get_summary_fn(config)

    def net(self, features, is_training):
        return self._net_fn(features, is_training, self.config.net_params)

    def loss(self, labels, net_out):
        return self._loss_fn(labels, net_out, self.config.loss_params)

    def predictions(self, net_out):
        return self._predictions_fn(net_out, self.config.predictions_params)

    def metrics(self, labels, net_out, mode):
        return self._metrics_fn(
            labels, net_out, mode, self.config.metrics_params)


    import tensorflow as tf


def normal_metrics(labels, net_out, mode, params):
    return {}


def accuracy_metrics(labels, net_out, mode, params):
    classes = tf.argmax(net_out, axis=1)
    classes = tf.cast(classes, dtype=tf.int32)
    if mode == tf.estimator.ModeKeys.TRAIN:
        correct = tf.cast(tf.equal(classes, labels), tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar('accuracy', accuracy)
    else:
        accuracy = tf.metrics.accuracy(labels, classes)
        tf.summary.scalar('accuracy', accuracy[1])
    metrics = {'accuracy': accuracy}
    return metrics


METRICS_MAP = {
    'normal': normal_metrics,
    'accuracy': accuracy_metrics,
}


def get_metrics_fn(config):
    if config.metrics not in METRICS_MAP:
        raise ValueError('{} is not a valid metrics type'.format(config.metrics))
    return METRICS_MAP[config.metrics]


    def lr_policy(self, global_step):
        return self._lr_policy_fn(global_step, self.config.lr_policy_params)

    def optimizer(self, lr):
        return self._optimizer_fn(lr, self.config.optimizer_params)

    def summary(self, features, labels, predictions, mode):
        return self._summary_fn(
            features, labels, predictions, mode, self.config.summary_params)

