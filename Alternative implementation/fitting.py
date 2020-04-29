epochs = 10
lr_list = [1e-3, 1e-4, 1e-5]
optimizer_list = ['SGD','Adam','Adamax']
df = pd.DataFrame(np.zeros((3, 9)))
accuracy_df = pd.DataFrame(np.zeros((epochs, 9)))
df.index = ['Time','Num of Paras','Accuracy']
colname_list = [str(i) + " lr = " + str(j) for i in optimizer_list for j in lr_list]
df.columns= colname_list
accuracy_df.columns= colname_list
i = 0

for opt in optimizer_list: 
  for lr in lr_list:
      print(str(opt) + " lr=" +str(lr))

      model = unet(lr,eval(opt))
      start = datetime.now()
      
      results = fit_model(epochs,50)
      end = datetime.now()
      time = end-start
      df.iloc[0,i] = time
      df.iloc[1,i] = model.count_params()
      df.iloc[2,i] = results.history['accuracy'][-1]
      results.history['accuracy']
      accuracy_df.iloc[:,i] = results.history['accuracy']

      i += 1

print(get_time())
