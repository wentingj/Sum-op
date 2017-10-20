import theano
from theano import tensor as T
import numpy as np
import sys
import sum_op
import keras.backend as K
import time

np.random.seed(12345)


def Sum_op(dimension):
    t = T.tensor3("t", dtype='float64')
    r = T.tensor3("r", dtype='float64')

    r = sum_op.Sum_op(dimension=dimension, keepdim=True)(t)
    loss = r.sum()
    gt = theano.grad(loss, t)

    f = theano.function([t], [r, gt])
    # theano.printing.pydotprint(f, outfile='sum_bw.png', var_with_name_simple=True)
    return f

def Sum_theano(dimension):
    t = T.tensor3("t", dtype='float64')
    r = T.tensor3("r", dtype='float64')
    keepdim = True

    r = K.sum(t, axis=dimension, keepdims=True)
    loss = r.sum()
    gt  = theano.grad(loss, t)

    f = theano.function([t], [r, gt])
    # theano.printing.pydotprint(f, outfile='sum_theano.png', var_with_name_simple=True)
    return f


if __name__ == '__main__':
    t = np.random.rand(8,6,10).astype(np.float64)
    dimension = 1

    f_op = Sum_op(dimension)
    f_theano = Sum_theano(dimension)

    tic = time.time()
    op_result = f_op(t)
    toc = time.time()
    print('op time: %.6f' %(toc - tic))
    
    tic = time.time()
    theano_result = f_theano(t)
    toc = time.time()
    print('theano time: %.6f' %(toc - tic))

    print op_result[1]
    print theano_result[1]
    print('\n====Compare sum result===================')
    assert np.allclose(op_result[0], theano_result[0])

    print('\n====Compare gradient inputs==============')
    assert np.allclose(op_result[1], theano_result[1])

