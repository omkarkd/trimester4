from numpy import log, dot, squeeze

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, log(Y_hat).T) + dot(1 - Y, log(1 - Y_hat).T))
    return squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = conver_pro_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()