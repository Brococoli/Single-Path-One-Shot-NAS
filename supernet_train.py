from supernet import *
import tensorflow as tf
import logging
from oneshot_nas_net import archs_choice
from utils.calculate_flops_params import get_flops_params
from utils.data import get_webface, get_cifar10
import math
from math import ceil
from copy import deepcopy

class Trainer(object):
    """supernet trainer"""
    def __init__(self, model, data, optimizer, **kwargs):
        super(Trainer, self).__init__()
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.flops_constant = kwargs.get('flops_constant', math.inf)
        self.params_constant = kwargs.get('params_constant', math.inf)

    def acc_func(y_true, y_pred):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def lr_plan(self, epoch):
        plan = [1e-3, 1e-4]
        lr = plan[min(len(plan)-1, epoch//45)]
        print('learning rate:', lr)
        tf.keras.backend.set_value(self.optimizer.lr, lr)

    def search_plan(self, epoch):
        return self.model.search_args
        """
        search_args = deepcopy(self.model.search_args)
        #warm up
        for idx, arg in enumerate(search_args):
            search_args[idx] = arg._replace(width_ratio=arg.width_ratio[:(epoch//3+2)],
                    expand_ratio=arg.expand_ratio[:(epoch//3+2)]) 
        return search_args
        """
        
    def train_step(self, images, labels, search_args):
        with tf.GradientTape() as g:
            logits = self.model(images, True, search_args=search_args)
            loss = self.loss_func(y_true = labels, y_pred = logits)
            loss += sum(self.model.losses)
        grads = g.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        acc = Trainer.acc_func(y_true = labels, y_pred = logits)
        return loss, acc

    def val_step(self, images, labels, search_args):
        logits = self.model(images, False, search_args=search_args)
        loss = self.loss_func(y_true = labels, y_pred = logits)
        loss += sum(self.model.losses)
        acc = Trainer.acc_func(y_true = labels, y_pred = logits)
        return loss, acc

    def get_archs(self,epoch):
        search_args = self.search_plan(epoch)
        return archs_choice_with_constant(self.data['imgs_shape'], 
            search_args, self.model.blocks_args, 
            flops_constant=self.flops_constant, params_constant=self.params_constant)

    def train(self, epochs, batch_size=128):
        train_ds = self.data['train_ds']
        train_num = self.data['train_num']
        val_ds = self.data['val_ds']
        val_num = self.data['val_num']
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(200)
        val_ds = val_ds.batch(batch_size).prefetch(200)

        epochs_probar = tf.keras.utils.Progbar(epochs)   
        for epoch in range(epochs):
            train_probar = tf.keras.utils.Progbar(ceil(train_num/batch_size))    #set value

            self.lr_plan(epoch)
            epochs_probar.update(epoch+1)
            print()
            for idx, (images, labels) in enumerate(train_ds):
                archs = self.get_archs(epoch) 
                loss, acc = self.train_step(images, labels, archs)
                train_probar.update(idx+1, values=[['accuracy', acc], ['loss', loss]])

            val_probar = tf.keras.utils.Progbar(ceil(val_num/batch_size))     #
            for idx, (images, labels) in enumerate(val_ds):
                archs = self.get_archs(epoch) 
                loss, acc = self.val_step(images, labels, archs)
                val_probar.update(idx+1, values=[['val_accuracy', acc], ['val_loss', loss]])


            self.model.save_weights('training_data/checkpoing/'+\
                'weights_{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.tf/'.format(epoch=epoch, val_loss=loss, val_accuracy=acc))
            logging.info('save the weights..')


def train():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


    logging.info('beging train...')

    model = get_nas_model('mobilenetv2-b0', blocks_type='mix', load_path='')
    logging.debug('get a nas model')

    data = get_webface()
    """
    data['train_ds'] = data['train_ds'].take(500)
    data['train_num'] = 500
    data['val_ds'] = data['val_ds'].take(500)
    data['val_num'] = 500
    """

    trainer = Trainer(model, data, optimizer=tf.keras.optimizers.Adam(1e-3), flops_constant=100)
    logging.debug('get a trainer')



    trainer.train(90, 128)


if __name__ == '__main__':
    import os,logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.get_logger().setLevel(logging.ERROR)
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    train()
