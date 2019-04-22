import os
import sys
import json

import numpy as np

# np.random.seed(0)
BIN_DIM = 8


def check_add(a, b):
    y = a + b
    return y if y > -127 and y < 128 else False


def gen_data(data_num):
    x = np.random.randint(-127, 128, size=(data_num, 2))
    gt = []
    for idx, (one, two) in enumerate(x):
        y = check_add(one, two)
        while y is False:
            new_x = np.random.randint(-127, 128, size=2)
            y = check_add(new_x[0], new_x[1])
            if y is not False:
                x[idx] = new_x
        gt.append(y)
    gt = np.array(gt)
    x = [np.int_(list(np.binary_repr(i, width=8))) for i in x.flatten()]
    x = np.array(x, dtype=np.int8).reshape(data_num, 2, 8).transpose(0, 2, 1)
    gt = [np.int_(list(np.binary_repr(i, width=8))) for i in gt.flatten()]
    gt = np.array(gt, dtype=np.int8).reshape(data_num, 8, 1)
    return x, gt


def gen_single_data():
    x = np.random.randint(-127, 128, size=2)
    if check_add(x[0], x[1]) is False:
        return gen_single_data()
    x = [np.int_(list(np.binary_repr(i, width=8))) for i in x.flatten()]
    x = np.array(x, dtype=np.int8).reshape(2, 8).transpose(1,0)
    y = np.array(list(np.binary_repr(i, width=8)), dtype=np.int8)
    y =  np.expand_dims(y, 1)
    return x, y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return np.multiply(x, 1.0 - x)


def tanh(z):
    ez = np.exp(z)
    enz = np.exp(-z)
    return (ez - enz)/ (ez + enz)


def tanh_derivative(x):
    return np.diag(1 - (np.tanh(x))**2)


def mse(y, gt):
    return (y - gt)**2


def mse_derivative(y, gt):
    return 2*(y - gt)


def softplus_derivative(y, gt):
    return (1 - 2*gt) / (1 + np.exp(-(1 - 2*gt) * y))


class RNN():
    def __init__(self, in_hidden_out_dim=[2, 16, 1], learning_rate=0.1, act_loss='tanh_softplus'):
        self.input_dim = in_hidden_out_dim[0]
        self.hidden_dim = in_hidden_out_dim[1]
        self.output_dim = in_hidden_out_dim[2]
        self.init_weights(in_hidden_out_dim)

        self.act_loss = act_loss
        if act_loss == 'tanh_softplus':
            self.actfunc = np.tanh
            self.actfunc_derivative = tanh_derivative
            self.lossfunc_derivative = softplus_derivative
        else:
            self.actfunc = sigmoid
            self.actfunc_derivative = sigmoid_derivative
            self.lossfunc_derivative = mse_derivative
        self.lossfunc = mse
        self.l_rate = learning_rate
        self.output_val = [np.zeros(in_hidden_out_dim[1])]
        self.acc_list = []
        self.accsize = 1000.0
        self.l_threshold = 0.1
    
    def init_weights(self, dimension):
        '''
        dimension[0] : input dimension
        dimension[1] : hidden dimension
        dimension[2] : output dimension
        '''
        self.u_weights = np.random.uniform(0, 1, (dimension[0], dimension[1]))
        self.v_weights = np.random.uniform(0, 1, (dimension[1], dimension[2]))
        self.w_weights = np.random.uniform(0, 0.5, (dimension[1], dimension[1]))
        # self.u_weights = np.random.random((dimension[0], dimension[1]))
        # self.v_weights = np.random.random((dimension[1], dimension[2]))
        # self.w_weights = np.random.random((dimension[1], dimension[1]))

    def last_acc(self, pred, ans):
        if len(self.acc_list) >= self.accsize:
            self.acc_list.pop(0)
        acc = True if np.array_equal(pred, ans) else False
        self.acc_list.append(acc)

    def get_accuracy(self):
        return np.sum(self.acc_list) / self.accsize

    def forward(self, step):
        x, y = gen_single_data()
        self.input_x = np.expand_dims(x, 1)
        gt = np.expand_dims(y, 1)
        pred_y = []
        total_error = 0
        self.h_values = [np.zeros(self.hidden_dim)]
        self.out_o_deltas = []
        for position in range(BIN_DIM):
            y = gt[-position - 1].T

            out_h = self.actfunc(np.dot(self.input_x[-position - 1],self.u_weights) + np.dot(self.h_values[-1],self.w_weights))
            if self.act_loss == 'tanh_softplus':
                out_o = sigmoid(np.dot(out_h,self.v_weights))
                self.out_o_deltas.append(self.lossfunc_derivative(out_o, y) * sigmoid_derivative(out_o))
            else:
                out_o = self.actfunc(np.dot(out_h,self.v_weights))
                self.out_o_deltas.append(self.lossfunc_derivative(out_o, y) * self.actfunc_derivative(out_o))
            error = self.lossfunc(out_o, y)
            total_error += np.abs(error[0])
            
            pred_y.append(np.round(out_o[0][0]))
            self.h_values.append(out_h.copy())
            
        pred_y = np.array(pred_y, dtype=np.int8)[::-1]
        ground_truth = gt.flatten()
        if step % 2000 == 0:
            print('-'*10, 'step', step, '-'*10)
            print('total error:', total_error)
            print('pred:', pred_y)
            print('true:', ground_truth)
        self.last_acc(pred_y, ground_truth)
        return pred_y, ground_truth

    def backward(self):
        u_weights_update = np.zeros_like(self.u_weights)
        v_weights_update = np.zeros_like(self.v_weights)
        w_weights_update = np.zeros_like(self.w_weights)
        future_out_h_delta = np.zeros(self.hidden_dim)
        
        for position in range(BIN_DIM):
            out_h = self.h_values[-position-1]
            pre_out_h = self.h_values[-position-2]
            
            out_o_delta = self.out_o_deltas[-position-1]
            out_h_delta = (future_out_h_delta.dot(self.w_weights.T) + out_o_delta.dot(self.v_weights.T)) * self.actfunc_derivative(out_h)
            
            v_weights_update += np.atleast_2d(out_h).T.dot(out_o_delta)
            w_weights_update += np.atleast_2d(pre_out_h).T.dot(out_h_delta)
            u_weights_update += self.input_x[position].T.dot(out_h_delta)
            
            future_out_h_delta = out_h_delta
            
        self.u_weights -= self._clipping(u_weights_update) * self.l_rate
        self.v_weights -= self._clipping(v_weights_update) * self.l_rate
        self.w_weights -= self._clipping(w_weights_update) * self.l_rate
        return np.mean(u_weights_update), np.mean(v_weights_update), np.mean(w_weights_update)

    def _clipping(self, weights):
        return weights * self.l_threshold / np.linalg.norm(weights) if np.linalg.norm(weights) > self.l_threshold else weights


def check_accuracy(pred, ans):
    error_count = 0
    for i in range(BIN_DIM):
        if pred[i] != ans[i]:
            error_count += 1
    return error_count / float(BIN_DIM) * 100


def main():
    # x, y = gen_data(100)
    rnn_net = RNN(learning_rate=0.1, act_loss='mse')
    acc_list = []
    update_rate = {'u': [], 'v': [], 'w': []}
    for count in range(20000):
        pred_y, gt = rnn_net.forward(count+1)
        u, v, w = rnn_net.backward()
        acc_list.append(check_accuracy(pred_y, gt))
        # acc_list.append(rnn_net.get_accuracy() * 100)
        update_rate['u'].append(u)
        update_rate['v'].append(v)
        update_rate['w'].append(w)
    print('last {} accuracy: {}%'.format(rnn_net.accsize, rnn_net.get_accuracy() * 100))
    
    with open('rnn_record.json', 'w') as f:
        json.dump({
            'accuracy': acc_list,
            'update_rate': update_rate
        }, f)


if __name__ == "__main__":
    main()
    
    sys.stdout.flush()
    sys.exit()