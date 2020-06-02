from oneshot_nas_net import *
import numpy as np


BLOCK_ARGS_A = [ #3398/10000
        BlockArgs(kernel_size=3, num_repeat=1, channels=32,
                  expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=2, channels=48,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=64,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=4, channels=64,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=96,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=160,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=1, channels=160,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
        ]

BLOCK_ARGS_B = [ #4760/10000
        BlockArgs(kernel_size=3, num_repeat=1, channels=32,
                  expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=2, channels=32,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=64,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=4, channels=64,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=96,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=160,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=1, channels=320,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
        ]


SEARCH_SPACE={
    'mobilenetv2':{
    'blocks_args': DEFAULT_BLOCKS_ARGS,
    'search_args': [
            SearchArgs(width_ratio=[1], kernel_size=[3], expand_ratio=[6] if i else [1]) for i in range(sum(arg.num_repeat for arg in DEFAULT_BLOCKS_ARGS))
        ]
    },
    'mobilenetv2-b0':{
    'blocks_args': DEFAULT_BLOCKS_ARGS,
    'search_args': [
            SearchArgs(width_ratio=np.arange(0.2,2.1,0.2), kernel_size=[3,5,7], expand_ratio=range(2,6) if i else [1]) for i in range(sum(arg.num_repeat for arg in DEFAULT_BLOCKS_ARGS))
        ]
    },
    'spos-b0':{
    'blocks_args': SPOS_BLOCKS_ARGS,
    'search_args': [
            SearchArgs(width_ratio=np.arange(0.2,2.1,0.2), kernel_size=[3,5,7], expand_ratio=range(2,6)) for i in range(sum(arg.num_repeat for arg in SPOS_BLOCKS_ARGS))
        ]
    },
    'spos-b1':{
    'blocks_args': BLOCK_ARGS_A,
    'search_args': [
            SearchArgs(width_ratio=np.arange(0.2,2.1,0.2), kernel_size=[3,5,7], expand_ratio=range(2,6)) for i in range(sum(arg.num_repeat for arg in BLOCK_ARGS_A))
        ]
    },
    'spos-b2':{
    'blocks_args': BLOCK_ARGS_B,
    'search_args': [
            SearchArgs(width_ratio=np.arange(0.2,2.1,0.2), kernel_size=[3,5,7], expand_ratio=range(2,6)) for i in range(sum(arg.num_repeat for arg in BLOCK_ARGS_B))
        ]
    },
}

def get_nas_model(type_name, load_path=None):
    model = SinglePathOneShot(dropout_rate=0.2,
                      drop_connect_rate=0.,
                      depth_divisor=8,
                      search_args=SEARCH_SPACE[type_name]['search_args'],
                      blocks_args=SEARCH_SPACE[type_name]['blocks_args'],
                      model_name='SinglePathOneShot',
                      activation='relu6',
                      use_se=False,
                      include_top=True,
                      pooling=None,
                      initializer = 'he_normal',
                      num_classes=10575,
                      weight_decay = 1e-4)
    if load_path:
        model.load_weights(load_path)
        print('weights from {} load successed'.format(load_path))
    return model

def test():
    model = get_nas_model('mobilenetv2-b0')
    model.save_weights('save.tf/')
    model = get_nas_model('mobilenetv2-b0', 'save.tf/')
    model = get_nas_model('spos-b0')

if __name__ == '__main__':
    #logging.debug('begin')
    test()