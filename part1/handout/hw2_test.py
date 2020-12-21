from autograder.hw2_autograder.test_conv import *
from autograder.hw2_bonus_autograder.test_conv2d import *
from autograder.hw2_bonus_autograder.test_pooling import *
from autograder.hw2_autograder.test_scanning import *
from autograder.hw2_autograder.test_cnn import *

from autograder.hw1_bonus_autograder.test_dropout import *
from autograder.hw1_bonus_autograder.test_adam import *
from autograder.hw1_bonus_autograder.test_batchnorm import *

# homework test step by step
# for debugging convenience

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # test_dropout_forward()
    # test_dropout_forward_backward()
    # test_linear_adam()
    # test_big_model_adam()
    test_linear_batchnorm_relu_forward_train()
    test_linear_batchnorm_relu_backward_train()
    test_big_linear_batchnorm_relu_train_eval()
    test_linear_batchnorm_relu_train_eval()
    # test_conv1d_forward()
    # test_conv1d_backward()
    # test_flatten()
    # test_simple_scanning_mlp()
    # test_distributed_scanning_mlp()
    # test_cnn_step()
    #
    # test_conv2d_forward()
    # test_conv2d_backward()
    # test_maxpool2d_forward()
    # test_maxpool2d_backward()
    #
    # test_avgpool2d_forward()
    # test_avgpool2d_backward()
if __name__ == '__main__':
    main()