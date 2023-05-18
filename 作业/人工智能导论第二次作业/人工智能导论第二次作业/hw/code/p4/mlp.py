import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

# input and output dimensions
input_dim = 784
output_dim = 10


def sigmoid(x):
    # TODO: forward pass of Sigmoid
    return 1/(1+np.exp(-x))


def softmax(x):
    # TODO: forward pass of Softmax
    # 计算指数
    exp_x = np.exp(x)
    
    # 沿着指定的轴计算指数的总和
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    # 对每个元素进行归一化处理
    softmax_output = exp_x / sum_exp_x
    
    return softmax_output

def sigmoid_grad(s):
    return s * (1 - s)

def calc_loss_and_grad(x, y, w1, w2, eval_only=False):
    """Forward Propagation and Backward Propagation.

    Given a mini-batch of images x, associated labels y, and a set of parameters, compute the
    cross-entropy loss and gradients corresponding to these parameters.

    :param x: images of one mini-batch.
    :param y: labels of one mini-batch.
    :param w1: weight parameters of layer 1.
    :param w2: weight parameters of layer 2.
    :param eval_only: if True, only return the loss and predictions of the MLP.
    :return: a tuple of (loss, dw2, dw1)
    """

    # Forward Propagation
    h1 = sigmoid(x.dot(w1))  # (B, H)
    y_hat = softmax(h1.dot(w2))  # (B, K)
    loss = - np.mean(np.sum(y * np.log(y_hat), axis=1))

    if eval_only:
        return loss, y_hat

    # Backward Propagation
    grad_y_hat = y_hat - y  # (B, K)
    grad_w2 = h1.T.dot(grad_y_hat) / x.shape[0]  # (H, K)
    grad_h1 = grad_y_hat.dot(w2.T)  # (B, H)
    grad_sigmoid = sigmoid_grad(h1)  # (B, H)
    grad_w1 = x.T.dot(grad_h1 * grad_sigmoid) / x.shape[0]  # (D, H)

    return loss, grad_w2, grad_w1


def train(train_x, train_y, val_x, val_y, test_x, text_y, args: argparse.Namespace):
    """Train the network.

    :param train_x: images of the training set.
    :param train_y: labels of the training set.
    :param test_x: images of the test set.
    :param text_y: labels of the test set.
    :param args: a dict of hyper-parameters.
    """

    #  randomly initialize the parameters (weights and biases)
    w1 = 2 * (np.random.random(size=(input_dim, args.hidden_dim)) / input_dim ** 0.5) - 1. / input_dim ** 0.5
    w2 = 2 * (np.random.random(size=(args.hidden_dim, output_dim)) / args.hidden_dim ** 0.5) - 1. / args.hidden_dim ** 0.5


    print('Start training:')
    print_freq = 100
    loss_curve = []

    for epoch in range(args.epochs):
        # train for one epoch
        print("[Epoch #{}]".format(epoch))

        # random shuffle dataset
        dataset = np.hstack((train_x, train_y)) 
        np.random.shuffle(dataset)
        train_x = dataset[:, :input_dim]
        train_y = dataset[:, input_dim:]

        n_iterations = train_x.shape[0] // args.batch_size

        for i in range(n_iterations):
            # load a mini-batch
            x_batch = train_x[i * args.batch_size: (i + 1) * args.batch_size, :] # B x 784
            y_batch = train_y[i * args.batch_size: (i + 1) * args.batch_size, :] # B x 10

            # compute loss and gradients
            loss, dw2, dw1 = calc_loss_and_grad(x_batch, y_batch, w1, w2)

            # update parameters
            w1 -= args.lr * dw1
            w2 -= args.lr * dw2

            loss_curve.append(loss)
            if i % print_freq == 0:
                print('[Iteration #{}/{}] [Loss #{:4f}]'.format(i, n_iterations, loss))

    # show learning curve
    plt.title('Training Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(loss_curve)), loss_curve)
    plt.show()

    # evaluate on the training set
    loss, y_hat = calc_loss_and_grad(train_x, train_y, w1, w2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(train_y, axis=1)
    accuracy = np.sum(predictions == labels) / train_x.shape[0]
    print('Top-1 accuracy on the training set', accuracy)

    # evaluate on the training set
    loss, y_hat = calc_loss_and_grad(val_x, val_y, w1, w2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(val_y, axis=1)
    accuracy = np.sum(predictions == labels) / val_x.shape[0]
    print('Top-1 accuracy on the validation set', accuracy)

    # evaluate on the test set
    loss, y_hat = calc_loss_and_grad(test_x, text_y, w1, w2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(text_y, axis=1)
    accuracy = np.sum(predictions == labels) / test_x.shape[0]
    print('Top-1 accuracy on the test set', accuracy)


def main(args: argparse.Namespace):
    # print hyper-parameters
    print('Hyper-parameters:')
    print(args)

    # load training set and test set
    raw_train_x, raw_train_y = utils.load_data("train")
    raw_dataset = np.hstack((raw_train_x, raw_train_y)) 
    train_dataset, val_dataset = utils.split_data(raw_dataset, holdout=0.1)
    train_x, train_y = train_dataset[:, :input_dim], train_dataset[:, input_dim:]
    val_x, val_y = val_dataset[:, :input_dim], val_dataset[:, input_dim:]
    test_x, text_y = utils.load_data("test")
    print('Dataset information:')
    print("training set size: {}".format(len(train_x)))
    print("validation set size: {}".format(len(val_x)))
    print("test set size: {}".format(len(test_x)))

    # check your implementation of backward propagation before starting training
    utils.check_grad(calc_loss_and_grad)

    # train the network and report the accuracy on the training and the test set
    train(train_x, train_y, val_x, val_y, test_x, text_y, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--hidden-dim', default=50, type=int,
                        help='hidden dimension of the Multilayer Perceptron')
    parser.add_argument('--lr', default=0.2, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
