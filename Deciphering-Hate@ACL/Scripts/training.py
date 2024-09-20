import architecture as m

def pipline(train_loader, val_loader, test_loader, epochs, lr_rate):

  # call the train function 
  print("Start Training DORA")
  model = m.train(train_loader, val_loader, epochs, lr_rate)

  # call test function
  actual_labels, pred_lables = m.evaluation(model, test_loader)

  return actual_labels, pred_lables
