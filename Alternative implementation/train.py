epochs = 10
def fit_model(epochs, batch_size):
  earlystopper = EarlyStopping(patience=5, verbose=1)
  checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
  results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, 
                callbacks=[earlystopper, checkpointer])
  return(results)
start = datetime.now()
  
model = unet(1e-3,Adam)
results = fit_model(epochs,50)



end = datetime.now()
time = end-start
