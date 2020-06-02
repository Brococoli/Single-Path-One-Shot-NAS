
import tensorflow as tf
from math import ceil



def get_flops_params(input_shape, blocks_args, search_args, ):
    params = flops = 0
    H, W, Ci = input_shape

    #type is mobilenet v2
    # first is stem : 3x3x3x32 conv
    H, W, Co = ceil(H/2), ceil(W/2), 32
    params += 3*3*Ci*Co
    flops += 3*3*Ci*Co * H * W
    Ci = Co
    assert sum(arg.num_repeat for arg in blocks_args) == len(search_args)

    num_blocks = 0
    for block_arg in blocks_args:
        K = block_arg.kernel_size
        S = block_arg.strides

        for _ in range(block_arg.num_repeat):
            E = search_args[num_blocks].expand_ratio
            F = ceil(search_args[num_blocks].width_ratio*block_arg.channels)
            F = max(F, 16)



            #MBConv: 1x1xCixCo conv
            Co = Ci*E
            if E != 1:
                param = 1*1*Ci*Co
                flop = param *H*W
                params += param
                flops += flop

            if S in [[2,2], 2]:
                H = ceil(H/2)
                W = ceil(W/2)

            #MBConv2: dwconv K*K*1*Ci
            Ci = Co
            param = K*K*1*Ci
            flop = param * H * W
            params += param
            flops += flop

            #MBConv3: conv 1*1*Ci*Co
            Ci = Co
            Co = F
            param = 1*1*Ci*Co
            flop = param * H*W
            params += param
            flops += flop

            num_blocks += 1
            Ci = Co
            S = 1
            #print(H,W,Co)

    Ci = Co
    Co = 1280
    #last head conv: 1280*Ci*1*1
    param = 1*1*Ci*Co
    flop = param * H*W
    params += param
    flops += flop

    #dense: 1280*10575
    flop = param = 1280*10575
    params += param
    flops += flop

            

    assert num_blocks == len(search_args)

    return flops, params

def test():
    import sys
    sys.path.append('../')
    import supernet
    import oneshot_nas_net
    from supernet import SEARCH_SPACE
    from oneshot_nas_net import archs_choice

    blocks_args = SEARCH_SPACE['spos-b2']['blocks_args']
    search_args = SEARCH_SPACE['spos-b2']['search_args']
    
    
    cnt = 0
    for i in range(10000):
        arch = archs_choice(search_args)
        flops, params = get_flops_params((112,96,3),blocks_args=blocks_args, search_args=arch)
        #print(flops/1000000)
        if flops/1000000 < 120:
           cnt+=1
    print(cnt)
   
    #print('mobilenetv2-b0',get_flops_params((112,96,3),blocks_args=blocks_args, search_args=arch))
    #print('mobilenetv2',get_flops_params((112,96,3),**SEARCH_SPACE['mobilenetv2']))
    #print('spos-b0', get_flops_params((112,96,3),**SEARCH_SPACE['spos-b0']))

if __name__ == '__main__':
    test()


