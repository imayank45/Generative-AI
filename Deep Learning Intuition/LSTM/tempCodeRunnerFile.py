
    ht, Ct = lstm_cell.forward(xt, ht_1, Ct_1)
    
    # Perform prediction
    predicted_index = vocabulary.index(target_word)
    
    # Loss computation and backpropagation (not i