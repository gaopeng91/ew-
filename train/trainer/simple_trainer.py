import tensorflow as tf
from tensorcv.train.trainer.trainer import Trainer
from tensorcv.train.hooks import CheckpointPerfactSaverHook
from tensorcv.logger import logger


class SimpleTrainer(Trainer):
    def get_model_fn(self, model):
        def model_fn(features, labels, mode):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            with tf.variable_scope('net'):
                net_out =  model.net(features, is_training)
            predictions = model.predictions(net_out)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions)

            loss = model.loss(labels, net_out)
            metrics = model.metrics(labels, net_out, mode)
            model.summary(features, labels, predictions, mode)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=metrics)

            if self.config.transfer_checkpoint:
                transfer_dict = {}
                transfer_params = self.config.transfer_params
                skip_names = transfer_params.get('skip_variables', [])
                for v in tf.trainable_variables():
                    is_skip = False
                    for skip_name in skip_names:
                        if skip_name in v.name:
                            is_skip = True
                            break
                    if not is_skip:
                        old_name = '/'.join(v.name.split('/')[1:]).split(':')[0]
                        transfer_dict[old_name] = 'net/' + old_name
                    else:
                        logger.info('Transfer skip variable {}'.format(v.name))
                tf.train.init_from_checkpoint(
                    self.config.transfer_checkpoint, transfer_dict)

            lr = model.lr_policy(tf.train.get_global_step())
            optimizer = model.optimizer(lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

            training_chief_hooks = [
                CheckpointPerfactSaverHook(
                    self.config.model_dir,
                    save_steps=self.config.model_save_steps)
            ]

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_chief_hooks=training_chief_hooks)

        return model_fn

