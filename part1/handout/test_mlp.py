from autograder.hw1_autograder.test_mlp import *
from autograder.hw1_autograder.test_mnist import *

def main():
    assert test_linear_forward()
    assert test_linear_backward()
    assert test_linear_relu_forward()
    assert test_linear_relu_backward()
    assert test_big_linear_relu_forward()
    assert test_big_linear_relu_backward()
    assert test_linear_relu_step()
    assert test_big_linear_relu_step()
    assert test_linear_xeloss_forward()
    assert test_linear_xeloss_backward()

    assert test_linear_momentum()
    assert test_big_linear_relu_xeloss_train_eval()
    assert test_big_linear_relu_xeloss_momentum()




if __name__ == '__main__':
    main()