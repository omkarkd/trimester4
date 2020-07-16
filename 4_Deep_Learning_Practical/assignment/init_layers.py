def init_layers(nn_architecture, seed=99):
    s(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_size']
        
        params_values['W' + str(layer_idx)] = randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = randn(
            layer_output_size, 1) * 0.1

    return params_values