from supernet import *
import tensorflow as tf
import logging
from oneshot_nas_net import archs_choice
from utils.calculate_flops_params import get_flops_params
from utils.data import get_webface, get_cifar10
import math
from math import ceil
from copy import deepcopy

class DistributeTrainer(object):
    """supernet DistributeTrainer"""
    def __init__(self, model, data, optimizer, strategy, **kwargs):
        super(DistributeTrainer, self).__init__()
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.strategy = strategy
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)

        self.flops_constant = kwargs.get('flops_constant', math.inf)
        self.params_constant = kwargs.get('params_constant', math.inf)
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        self.train_loss = tf.keras.metrics.Sum(name='train_loss')
        self.val_loss = tf.keras.metrics.Sum(name='val_loss')


    def compute_loss(self, y_true, y_pred):
        loss = tf.reduce_sum(self.loss_func(y_true = y_true, y_pred = y_pred)) * (1. / self.batch_size)
        loss += sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync
        return loss

    def acc_func(y_true, y_pred):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def lr_plan(self, epoch):
        plan = [1e-3, 1e-4]
        lr = plan[min(len(plan)-1, epoch//45)]
        print('learning rate:', lr)
        tf.keras.backend.set_value(self.optimizer.lr, lr)

    def search_plan(self, epoch):
        search_args = deepcopy(self.model.search_args)
        #warm up
        for idx, arg in enumerate(search_args):
            search_args[idx] = arg._replace(width_ratio=arg.width_ratio[:(epoch//3+2)],
                    expand_ratio=arg.expand_ratio[:(epoch//3+2)]) 
            return search_args

    def train_step(self, images, labels, search_args):
        with tf.GradientTape() as g:
            logits = self.model(images, True, search_args=search_args)
            loss = self.compute_loss(y_true = labels, y_pred = logits)
        grads = g.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        acc = self.train_acc_metric(labels, logits)
        return loss, acc

    def val_step(self, images, labels, search_args):
        logits = self.model(images, False, search_args=search_args)
        loss = self.loss_func(y_true = labels, y_pred = logits)
        assert loss.shape == []
        loss += sum(self.model.losses)
        acc = self.val_acc_metric(labels, logits)
        return loss, acc

    def get_archs(self,epoch):
        search_args = self.search_plan(epoch)
        return archs_choice_with_constant(self.data['imgs_shape'], 
                search_args, self.model.blocks_args, 
                flops_constant=self.flops_constant, params_constant=self.params_constant)

    def train(self, epochs, batch_size=128):
        self.batch_size = batch_size
        train_ds = self.data['train_ds']
        train_num = self.data['train_num']
        val_ds = self.data['val_ds']
        val_num = self.data['val_num']
        train_ds = self.strategy.experimental_distribute_dataset(train_ds.shuffle(1000).batch(batch_size))
        val_ds = self.strategy.experimental_distribute_dataset(val_ds.batch(batch_size))

        epochs_probar = tf.keras.utils.Progbar(epochs)   
        for epoch in range(epochs):
            train_probar = tf.keras.utils.Progbar(ceil(train_num/batch_size), stateful_metrics=['acc'])    #set value
            self.lr_plan(epoch)
            epochs_probar.update(epoch+1)
            print()

            total_loss = 0.0
            for idx, (images, labels) in enumerate(train_ds):
                archs = self.get_archs(epoch) 
                per_replica_loss, acc = self.strategy.experimental_run_v2(self.train_step, args=(images, labels, archs))
                total_loss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

                train_probar.update(idx+1, values=[['accuracy', acc], ['loss', total_loss/(idx+1)]])

            val_probar = tf.keras.utils.Progbar(ceil(val_num/batch_size), stateful_metrics=['val_accuracy', 'val_loss'])     #
            for idx, (images, labels) in enumerate(val_ds):
                archs = self.get_archs(epoch) 
                loss, acc = self.strategy.experimental_run_v2(self.train_step, args=(images, labels, archs))
                val_probar.update(idx+1, values=[['val_accuracy', acc], ['val_loss', loss]])


            self.model.save_weights('training_data/checkpoing/'+\
                    'weights_{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.tf/'.format(epoch=epoch, val_loss=loss, val_accuracy=acc))
            logging.info('save the weights..')

            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_status()


def train():
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


    devices = ['/device:GPU:{}'.format(i) for i in [0,2]]

    strategy = tf.distribute.MirroredStrategy(devices)
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    data = get_webface()

    with strategy.scope():
        model = get_nas_model('mobilenetv2-b0', blocks_type='nomix')

        trainer = DistributeTrainer(model, data, optimizer=tf.keras.optimizers.Adam(1e-3), strategy=strategy, flops_constant=120)
        logging.info('beging train...')
        trainer.train(90, 256)



if __name__ == '__main__':
    import os,logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    tf.get_logger().setLevel(logging.ERROR)
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    train()

