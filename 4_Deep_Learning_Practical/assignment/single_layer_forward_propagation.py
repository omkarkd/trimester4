def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation='relu'):
    Z_curr = dot(W_curr, A_prev) + b_curr

    if activation is 'relu':
        activation_func = relu
    elif activation isd 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr