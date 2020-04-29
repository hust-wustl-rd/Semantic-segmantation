def get_performance():
  print("Time: " + str(time))
  print("number of parameters: " + str(model.count_params()))
  print("Accuracy: " + str(results.history['accuracy'][-1]))
  plt.plot(np.arange(1,len(results.epoch)+1),results.history['accuracy'])
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
