import numpy as np
import scipy.io
import h5py

DATA_PATH = 'deepsea_train/'

## TRAIN DATA ###
print("Loaging training data set...")
train_data_raw = h5py.File(DATA_PATH+'train.mat')

x_train = train_data_raw['trainxdata']
y_train = train_data_raw['traindata']
print("training data set [DONE]")


for i,n in enumerate(range(0,1000,100)):
    fn = "x_train_part_{}".format(i)
    yfn = "y_train_part_{}".format(i)
    
    print("\nSaving Part {} ....".format(i))
    np.save(DATA_PATH+fn, x_train[n:n+100])
    np.save(DATA_PATH+yfn, y_train[n:n+100])
    print("{} [Done]".format(fn))
    print("{} [Done]".format(yfn))


