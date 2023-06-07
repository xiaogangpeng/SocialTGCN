import os

import numpy as np

# loading data

train_data = []

data_path = 'data/MI-Motion/'
use = [0,1,3,4,5,7,8,9,10,11,12,13, 14,15,16, 17,18,19]  # get 18 body joints

for ii in range(5):
    test_data = []

    motion_list_train = []
    motion_list_test= []  

    # 0 is park, 1 is street, 2 is indoor, 3 is special locations, 4 is complex crowd
    # index for test sample
    if ii == 0:
        sub_dir = 'S0'  
        index = ['27', '28', '45', '48', '49']
    if ii == 1:
        sub_dir = 'S1'
        index = ['3', '18', '31', '42', '43']
    if ii == 2:
        sub_dir = 'S2'
        index = ['1','4', '12', '27', '48']
    if ii == 3:
        sub_dir = 'S3'
        index = ['6', '7', '23', '25', '29']
    if ii == 4:
        sub_dir = 'S4'
        index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']     


    for each in os.listdir(data_path+sub_dir):
        import re
        filename = sub_dir + '_'+'12'+'.npy'
        each_id = re.split('[_.]', each)[1]
        if each_id in index:
            motion_list_test.append(each)
        else:
            motion_list_train.append(each)

    scene_length = len(motion_list_train)
    for i in range(scene_length):
        npy_data = np.load(data_path + sub_dir+'/'+motion_list_train[i])
        for j in range(0, npy_data.shape[1], 10):
            if j + 150 < npy_data.shape[1]:
                motion = npy_data[:,j:j + 150,:,:]  * 1.8 / 90 # scale
                motion[:, :, :, 0] = motion[:, :, :, 0] - np.mean(motion[:, :, :, 0])   # centering
                motion[:, :, :, 1] = motion[:, :, :, 1] - np.mean(motion[:, :, :, 1])
                train_data.append(motion[:,::3,use,:])  # down sample to 25 fps

    scene_length = len(motion_list_test)
    for i in range(scene_length):
        npy_data = np.load(data_path + sub_dir+'/'+motion_list_test[i])
        for j in range(0, npy_data.shape[1], 75):
            if j + 225 < npy_data.shape[1]:

                motion = npy_data[:,j:j + 225,:,:] * 1.8 / 90  # scale
                motion[:, :, :, 0] = motion[:, :, :, 0] - np.mean(motion[:, :, :, 0])   # centering
                motion[:, :, :, 1] = motion[:, :, :, 1] - np.mean(motion[:, :, :, 1])
                test_data.append(motion[:,::3,use,:])  # down sample to 25 fps
    
    test_data_ = np.array(test_data)
    np.save(data_path+'data_test_'+sub_dir+'.npy', test_data_)
    print(f"test{sub_dir}: {len(test_data_)}")

train_data_ = np.array(train_data)
np.save(data_path+'data_train.npy', train_data_)
print(f"train: {len(train_data_)}")
