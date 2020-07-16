from numpy import dot, sum as s

def single_layer_backward_progation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    m = A_prev.shape[1]

    if activation is 'relu':
        backward_activation_func = relu_backward
    elif activation is 'sigmoid':
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = dot(dZ_curr, A_prev.T) / m
    db_curr = s(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = dot(W_curr, dW_curr, db_curr)