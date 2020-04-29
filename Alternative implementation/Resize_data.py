name_list_train = open('./train_list/train.txt','r')
lines_train = name_list_train.readlines()
X_train = np.zeros((len(lines_train), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(lines_train), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n in range(len(lines_train)):
      file_path = X_PATH + str(lines_train[n][0:-1])+ ".jpg"
      img_x = imread(file_path)[:,:,:IMG_CHANNELS]
      img_x = resize(img_x, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      X_train[n] = img_x
      file_path = Y_PATH + str(lines_train[n][0:-1])+ ".png"
      img_y = imread(file_path)[:,:,:1]
      img_y = resize(img_y, (IMG_HEIGHT, 1), mode='constant', preserve_range=True)
    
      Y_train[n] = img_y
print(get_time())


name_list_test = open('./train_list/val.txt','r')
lines_test = name_list_test.readlines()
X_test = np.zeros((len(lines_test), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(lines_test), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n in range(len(lines_test)):
      file_path = X_PATH + str(lines_test[n][0:-1])+ ".jpg"
      img_x = imread(file_path)[:,:,:IMG_CHANNELS]
      img_x = resize(img_x, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      X_test[n] = img_x
      file_path = Y_PATH + str(lines_test[n][0:-1])+ ".png"
      img_y = imread(file_path)[:,:,:1]
      img_y = resize(img_y, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      Y_test[n] = img_y
print("done")
print(get_time())
