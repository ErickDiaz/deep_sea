import numpy as np
import scipy.io
import h5py

DATA_PATH = 'deepsea_train/'

## TRAIN DATA ###
print("Loaging training data set...")
train_data_raw = h5py.File(DATA_PATH+'train.mat')

x_train = train_data_raw['trainxdata']
y_train = train_data_raw['traindata']

x_train = np.transpose(x_train, (2, 1, 0)) # shape = (4400000,  4, 1000)
y_train = np.transpose(y_train, (1, 0)) # shape = (4400000, 919)

print("training data set [DONE]")

split = 400000
for i,n in enumerate(range(0,4400000,split)):
    fn = "x_train_part_{}".format(i)
    yfn = "y_train_part_{}".format(i)
    
    print("\nSaving Part {} ....[{}:{}]".format(i, n, n+split))
    np.save(DATA_PATH+fn, x_train[n:n+split])
    np.save(DATA_PATH+yfn, y_train[n:n+split])
    print("{} [Done]".format(fn))
    print("{} [Done]".format(yfn))