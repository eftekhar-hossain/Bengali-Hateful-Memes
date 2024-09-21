import architecture as m

def pipline(train_loader, valid_loader, test_loader, task, model_path, n_heads, class_weights, epochs, lr_rate):

  # call the train function 
  model = m.train(train_loader, valid_loader, task, model_path, n_heads, class_weights, epochs, lr_rate)

  # call test function
  actual_labels, pred_lables = m.evaluation(model_path, task, model, test_loader)

  return actual_labels, pred_lables
