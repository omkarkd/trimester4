def sigmoid(Z):
    return 1/(1+exp(-Z))

def relu(Z):
    return maximum(0, Z)

def sigmoid_backward(dA, Z):
    return dA * sigmoid(Z) * (1 - sigmoid(Z))

def relu_backward(dA, Z):
    dZ = array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ