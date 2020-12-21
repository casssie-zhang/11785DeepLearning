from autograder.hw3_autograder.test_functional import *
from autograder.hw3_autograder.test_util import *
from autograder.hw3_autograder.test_rnn import *
from autograder.hw3_autograder.test_gru import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    test_rnn_unit_forward()
    test_rnn_unit_backward()
    test_concat_forward()
    test_concat_backward()
    test_pack_sequence_forward()
    test_rnn_unit_forward()
    test_gru_unit_forward()