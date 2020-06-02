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
    def __init__(self,  data, model, flops_constant, params_constant):
        super(Searcher, self).__init__()
        self.data = data
        self.model = model
        self.flops_constant = flops_constant
        self.params_constant = params_constant

    def set_update_bn(layer, inference_update_stat):
        if hasattr(layer, 'layers'):
            for ll in layer.layers:
                Searcher.set_update_bn(ll, inference_update_stat)
        else:
            if isinstance(layer, SuperBatchNormalization):
                layer.inference_update_stat = inference_update_stat
                logging.debug('set layer bn: %s' % str(layer))

    def update_bn(self, arch):
        #assert tf.reduce_mean(model.layers[1].layers[1].layers[1].bn.moving_mean) == 0
        Searcher.set_update_bn(self.model, inference_update_stat=True)
        probar = tf.keras.utils.Progbar(200)
        logging.info('begin update bn...')
        #print('update bn...')
        for idx, (imgs, labels) in enumerate(self.data['train_ds'].batch(128).take(200)):
            assert imgs.shape[1:] == (32,32,3)
            self.model(imgs, False, arch)
            probar.update(idx+1)

        Searcher.set_update_bn(model, inference_update_stat=False)
        #assert tf.reduce_mean(model.layers[1].layers[1].layers[1].bn.moving_mean) != 0

    def get_accuracy(self, arch):
        probar = tf.keras.utils.Progbar(200, stateful_metrics=['val_acc'])
        logging.info('begin cal val acc...')
        acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy('val_acc')
        for idx, (imgs, labels) in enumerate(self.data['val_ds'].batch(128).take(200)):
            assert imgs.shape[1:] == (32,32,3)
            logits = model(imgs, False, arch)
            acc = acc_metrics(labels, logits)
            probar.update(idx+1, values=[['val_acc', acc]])
        return acc

    def random_search(self, search_iters):
        result = []
        probar = tf.keras.utils.Progbar(search_iters)
        while len(result) < search_iters:
            arch = archs_choice_with_constant(self.data['imgs_shape'], 
                                self.model.search_args, self.model.blocks_args, 
                                flops_constant=self.flops_constant, params_constant=self.params_constant)
            flops, params = get_flops_params(self.data['imgs_shape'], search_args=arch, blocks_args=self.model.blocks_args)
            flops_score, params_score = flops / 1000000 / self.flops_constant, params / 1000000 / self.params_constant
            combined_score = 0.5 * flops_score + 0.5* params_score
            
            

            print("Target size + 1, with normalized score: {}".format(combined_score))
            print('flops: %.4f, params: %.4f' % (flops, params))
            arch_list = [list(i) for i in arch]
            print('archs:', arch_list)

            #update_bn
            #tic = time.time()
            self.update_bn(arch)
            #print("BN statistics updated. Time used: {}".format(time.time() - tic))
        
            # get validation accuracy
            #tic = time.time()
            val_acc = self.get_accuracy(arch)

            #print("Validation accuracy evaluated. Time used: {}".format(time.time() - tic))
                    
            # update the list of best networks
            result.append( dict(flops=flops, params=params, flops_score=flops_score, params_score=params_score,
                                combined_score=combined_score, val_acc=val_acc, arch = arch_list))

            if i % 10 == 0:
                np.save('arch_search.npy', result, allow_pickle=True)
                logging.debug('save arch number: %d'% len(result))
            probar.add(1)

        return result

def search():
    type_name = 'mobilenetv2-b0'
    load_path = ''
    model = get_nas_model(type_name, load_path)

    data = get_cifar10()

    flops_constant = 80*1.2
    params_constant = 15*1.2
    seacher = Searcher(data, model, flops_constant=flops_constant, params_constant=params_constant)
    result = seacher.random_search(1000)
    np.save('arch_search.npy', allow_pickle=True)


if __name__ == '__main__':
    search()
