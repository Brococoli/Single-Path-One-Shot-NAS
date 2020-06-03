from oneshot_nas_blocks import SuperBatchNormalization
import logging
import tensorflow as tf
from supernet_train import Trainer
from utils.calculate_flops_params import get_flops_params
from utils.data import get_webface, get_cifar10
from oneshot_nas_net import *
from supernet import *

class Searcher(object):
    """ Search the best arch"""
    def __init__(self,  data, model, flops_constant, params_constant, **kwargs):
        super(Searcher, self).__init__()
        #prefetch imgs
        update_bn_imgs = kwargs.get('update_bn_imgs', 20000)
        self.batch_size = kwargs.get('batch_size', 128)
        data['train_ds'] = data['train_ds'].take(update_bn_imgs).batch(self.batch_size).prefetch(100)
        data['train_num'] = update_bn_imgs
        data['val_ds'] = data['val_ds'].batch(self.batch_size).prefetch(100)
        
        
        self.data = data
        self.model = model
        self.flops_constant = flops_constant
        self.params_constant = params_constant
        

    def set_update_bn(layer, inference_update_stat):
        if hasattr(layer, 'layers'):
            for ll in layer.layers:
                Searcher.set_update_bn(ll, inference_update_stat)
        else:
            if isinstance(layer, (SuperBatchNormalization, tf.keras.layers.BatchNormalization)):
                if isinstance(layer, SuperBatchNormalization):
                    layer.inference_update_stat = inference_update_stat
                    logging.debug('set layer bn: %s' % str(layer))

                if inference_update_stat == True:
                    for weight in layer.weights:
                        if 'moving_var' in weight.name:
                            weight.assign(tf.ones(weight.shape))
                            logging.debug('set moving var ones: %s' % str(layer))
                        elif 'moving_mean' in weight.name:
                            weight.assign(tf.zeros(weight.shape))
                            logging.debug('set moving mean zeros: %s' % str(layer))

    def update_bn(self, arch):
        #assert tf.reduce_mean(model.layers[1].layers[1].layers[1].bn.moving_mean) == 0
        Searcher.set_update_bn(self.model, inference_update_stat=True)
        probar = tf.keras.utils.Progbar(ceil(self.data['train_num']/self.batch_size))
        #print('update bn...')
        for idx, (imgs, labels) in enumerate(self.data['train_ds']):
            self.model(imgs, True, arch)
            probar.update(idx+1)

        Searcher.set_update_bn(self.model, inference_update_stat=False)
        #assert tf.reduce_mean(model.layers[1].layers[1].layers[1].bn.moving_mean) != 0

    def get_accuracy(self, arch):
        probar = tf.keras.utils.Progbar(ceil(self.data['val_num']/self.batch_size), stateful_metrics=['val_acc'])
        acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy('val_acc')
        for idx, (imgs, labels) in enumerate(self.data['val_ds']):
            logits = self.model(imgs, False, arch)
            acc = acc_metrics(labels, logits)
            probar.update(idx+1, values=[['val_acc', acc]])
        return acc

    def random_search(self, search_iters=1000):
        result = []
        probar = tf.keras.utils.Progbar(search_iters)
        for i in range(search_iters):
            arch = archs_choice_with_constant(self.data['imgs_shape'], 
                                self.model.search_args, self.model.blocks_args, 
                                flops_constant=self.flops_constant, params_constant=self.params_constant)
            flops, params = get_flops_params(self.data['imgs_shape'], search_args=arch, blocks_args=self.model.blocks_args)
            flops_score, params_score = flops / 1000000 / self.flops_constant, params / 1000000 / self.params_constant
            combined_score = 0.5 * flops_score + 0.5* params_score
            

            logging.info("Target size + 1, with normalized score: {}".format(combined_score))
            logging.info('flops: %.4f, params: %.4f' % (flops, params))
            arch_list = [list(i) for i in arch]
            logging.info('archs: %s' % str(arch_list))

            logging.info('begin update bn...')
            self.update_bn(arch)
            logging.info('update bn done...')
        
            logging.info('begin cal val acc...')
            val_acc = self.get_accuracy(arch)
            logging.info('cal val acc done...')
                    
            result.append( dict(flops=flops, params=params, flops_score=flops_score, params_score=params_score,
                                combined_score=combined_score, val_acc=val_acc, arch = arch_list))

            if i % 10 == 0:
                np.save('arch_search.npy', result, allow_pickle=True)
                logging.info('save arch number: %d'% len(result))
            probar.add(1)

        return result

def search():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    type_name = 'mobilenetv2-b0'
    load_path = ''
    model = get_nas_model(type_name, blocks_type='nomix', load_path=load_path)

    data = get_cifar10()

    flops_constant = 80*1.2
    params_constant = 15*1.2

    seacher = Searcher(data, model, flops_constant=flops_constant, params_constant=params_constant)

    logging.info('random search begining...')
    result = seacher.random_search(1000)

    np.save('arch_search.npy', allow_pickle=True)
    logging.info('random search done')

if __name__ == '__main__':
    import os,logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    search()
