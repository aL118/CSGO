import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # single GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # use multiple GPUs

import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(["GPU:0"])
# strategy = tf.distribute.MirroredStrategy(["GPU:0","GPU:1","GPU:2", "GPU:3"])
print('\nnumber of devices using for training: {}'.format(strategy.num_replicas_in_sync))

import numpy as np
import time
import datetime
import pickle
import random
import h5py

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add, ReLU, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, concatenate, Input, AveragePooling2D, TimeDistributed, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications import EfficientNetB0
from random import sample

import sys
sys.path.insert(0,'D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning')
from config import *

# inputs
batch_size = 1 # this is total batchsize using all GPUs, so make divisible by num_gpus!
l_rate = 0.0001

# training data location
file_name_stub = 'dm_july2021_expert_' # dm_july2021_ aim_july2021_expert_ dm_july2021_expert_
# file_name_stub = 'dm_6nov_aim_' 
orig_folder_name = 'D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\dataset_dm_expert_dust2\\' 
my_folder_name = 'D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\hdfs\\' 
starting_num = 1 # lowest file name to use in training
highest_num = 30 # highest file name to use in training 4000, 5500, 190, 45, 10

# whether to save model if training and where
model_name = 'test_model'
save_dir = 'D:\\lyzheng\\projects\\angela\\Counter-Strike_Behavioural_Cloning\\my_models\\'
SAVE_MODEL = True

# whether to resume training from a previous model
IS_LOAD_WEIGHTS_AND_MODEL=False
weights_name = 'test_model_1'

# which subselection of dataset to train on
IS_SUBSELECT = False
SUB_PROB = 0.4
SUB_TYPE = 'ak' # ak or akm4 or all
OVERSAMPLE_LOWFREQ_REGION=False

# where are the metadata .npy files? only needed if subselecting
curr_vars_folder = '/mfs/TimPearce/01_csgo/03_currvars/'
if file_name_stub == 'dm_july2021_expert_':
    curr_vars_stub = 'currvarsv2_dm_july2021_expert_'
else:
    curr_vars_stub = 'currvarsv2_dm_july2021_'


start_time=time.time()
with strategy.scope():
    # useful tutorial for building, https://keras.io/getting-started/functional-api-guide/
    print('-- building model from scratch --')

    base_model = EfficientNetB0(weights='imagenet',input_shape=(input_shape[1:]),include_top=False,drop_connect_rate=0.2)
    base_model.trainable = True

    intermediate_model= Model(inputs=base_model.input, outputs=base_model.layers[161].output)
    intermediate_model.trainable = True

    input_1 = Input(shape=input_shape,name='main_in')
    x = TimeDistributed(intermediate_model)(input_1)


    x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=False,return_sequences=True)(x)

    x = TimeDistributed(Flatten())(x)

    # 2) set up auxillary input,  which can have previous actions, as well as aux info like health, ammo, team
    aux_input = Input(shape=(int(ACTIONS_PREV*(aux_input_length))),name='aux_in')

    # 3) add shared fc layers
    dense_5 = x

    # 4) set up outputs, sepearate outputs will allow seperate losses to be applied
    output_1 = TimeDistributed(Dense(n_keys, activation='sigmoid'))(dense_5)
    output_2 = TimeDistributed(Dense(n_clicks, activation='sigmoid'))(dense_5)
    output_3 = TimeDistributed(Dense(n_mouse_x, activation='softmax'))(dense_5) # softmax since mouse is mutually exclusive
    output_4 = TimeDistributed(Dense(n_mouse_y, activation='softmax'))(dense_5) 
    output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5) 
    output_all = concatenate([output_1,output_2,output_3,output_4,output_5], axis=-1)


    # 5) finish model definition
    model = Model(input_1, output_all)

    print(model.summary())

    # loss to minimise
    def custom_loss(y_true, y_pred):
        # y_true is shape [n_batch, n_timesteps, n_keys+n_clicks+n_mouse_x+n_mouse_y+n_reward+n_advantage]
        # where n_reward and n_advantage must =1
        # y_pred is shape [n_batch, n_timesteps, n_keys+n_clicks+n_mouse_x+n_mouse_y+n_val]
        # we'll use y_true to send in reward and eventually original advantage fn (or could recompute this?)

        # wasd keys
        loss1a = losses.binary_crossentropy(y_true[:,:,0:4], 
                                            y_pred[:,:,0:4])
        # space key
        loss1b = losses.binary_crossentropy(y_true[:,:,4:5], 
                                            y_pred[:,:,4:5])
        # reload key
        loss1c = losses.binary_crossentropy(y_true[:,:,n_keys-1:n_keys], 
                                            y_pred[:,:,n_keys-1:n_keys])

        # weapon switches, 1,2,3
        loss1d = losses.binary_crossentropy(y_true[:,:,n_keys-4:n_keys-1], 
                                            y_pred[:,:,n_keys-4:n_keys-1])

        # all other keys
        # loss1d = losses.binary_crossentropy(y_true[:,:,5:n_keys-1], 
        #                                     y_pred[:,:,5:n_keys-1])
        # left click
        loss2a = losses.binary_crossentropy(y_true[:,:,n_keys:n_keys+1], 
                                            y_pred[:,:,n_keys:n_keys+1])
        # right click
        loss2b = losses.binary_crossentropy(y_true[:,:,n_keys+1:n_keys+n_clicks], 
                                            y_pred[:,:,n_keys+1:n_keys+n_clicks])
        # mouse move x
        loss3 = losses.categorical_crossentropy(y_true[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x], 
                                                y_pred[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x])
        # mouse move y
        loss4 = losses.categorical_crossentropy(y_true[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y], 
                                                y_pred[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y])

        # critic loss -- measuring between consecutive time steps
        #  = ((reward_t + gamma  v_t+1) - v_t)^2
        # can't really decide whether to use reward_t or reward_t+1, but guess it doesn't matter too much
        loss_crit = 10*losses.MSE(y_true[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                           + GAMMA*y_pred[:,1:,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                           ,y_pred[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1])

        return K.concatenate([loss1a, loss1b, loss1c, loss2a, loss3, loss4, loss_crit])
        # return K.concatenate([loss1a, loss2a, loss3, loss4])




    # metrics for each part of interest - useful for debugging
    def wasd_acc(y_true, y_pred):
        return keras.metrics.binary_accuracy(y_true[:,:,0:4], y_pred[:,:,0:4])

    def j_acc(y_true, y_pred): # other keys, space, ctrl, shift, 1,2,3, r
        return keras.metrics.binary_accuracy(y_true[:,:,4:5], y_pred[:,:,4:5])

    def oth_keys_acc(y_true, y_pred): # other keys, space, ctrl, shift, 1,2,3, r
        return keras.metrics.binary_accuracy(y_true[:,:,5:n_keys], y_pred[:,:,5:n_keys])

    def Lclk_acc(y_true, y_pred):
        return keras.metrics.binary_accuracy(y_true[:,:,n_keys:n_keys+1], y_pred[:,:,n_keys:n_keys+1],threshold=0.5)
        # relative to proportion that don't fire 
        # return keras.metrics.binary_accuracy(y_true[:,n_keys:n_keys+1], y_pred[:,n_keys:n_keys+1],threshold=0.5) - (1 - keras.backend.mean(keras.backend.greater(y_true[:,n_keys:n_keys+1], 0.5)))

    def Rclk_acc(y_true, y_pred):
        return keras.metrics.binary_accuracy(y_true[:,:,n_keys+1:n_keys+n_clicks], y_pred[:,:,n_keys+1:n_keys+n_clicks],threshold=0.5)

    def m_x_acc(y_true, y_pred):
        return keras.metrics.categorical_accuracy(y_true[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x], 
                                                  y_pred[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x])
    def m_y_acc(y_true, y_pred):
        return keras.metrics.categorical_accuracy(y_true[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y], 
                                                  y_pred[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y])

    def crit_mse(y_true, y_pred):
        return 100*losses.MSE(y_true[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                                               + GAMMA*y_pred[:,1:,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                                               ,y_pred[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1])


    def w_1(y_true, y_pred):
        return keras.backend.mean(keras.backend.greater(y_true[:,:,0], 0.5))

    def no_fire(y_true, y_pred):
        return 1 - keras.backend.mean(keras.backend.greater(y_true[:,:,n_keys:n_keys+1], 0.5))

    def m_x_0(y_true, y_pred):
        return keras.backend.mean(keras.backend.greater(y_true[:,:,n_keys+n_clicks+int(np.floor(n_mouse_x/2))], 0.5))

    def m_y_0(y_true, y_pred):
        return keras.backend.mean(keras.backend.greater(y_true[:,:,n_keys+n_clicks+n_mouse_x+int(np.floor(n_mouse_y/2))], 0.5))


    opt = optimizers.Adam(learning_rate=l_rate)
    # model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_x_0,m_y_acc,m_y_0])
    model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_y_acc,wasd_acc,crit_mse])
    # model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_y_acc,wasd_acc])
    print('successfully compiled model')

# data generator
class DataGenerator(keras.utils.Sequence):
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end() # run this once to start

    def __len__(self):
        # the number of batches per epoch - how many times are we calling this generator altogether
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # generate one batch of data, index is the batch number

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        my_list_IDs_temp = [self.list_IDs[k] for k in range(0,min(90,len(self.list_IDs)))]

        # X1, y1 = self.__data_generation(list_IDs_temp, orig_folder_name, "dm_july2021_expert_")
        X, y = self.__data_generation(my_list_IDs_temp, my_folder_name, "test_")
        # sample half from each dataset
        # orig = sample(range(0,96),48)
        # X = np.array([[X1[0][i] if i in orig else X2[0][i] for i in range(0,96)]])
        # y = np.array([[y1[0][i] if i in orig else y2[0][i] for i in range(0,96)]])
        return X, y

    def on_epoch_end(self):
        # updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        # could do subsampling at this stage, by 
        # using ID in format 'filenum-framenum-containkillevent'

    def __data_generation(self, list_IDs_temp, folder_name, stub_name):

        # set up empty arrays to fill
        x_shape = list(input_shape)
        x_shape.insert(0,self.batch_size)
        # y_shape = [self.batch_size,n_keys+n_clicks+n_mouse_x+n_mouse_y]
        y_shape = [self.batch_size,N_TIMESTEPS,n_keys+n_clicks+n_mouse_x+n_mouse_y+1+1] # add 1 for reward, 1 for adv

        X = np.empty(x_shape)
        y = np.empty(y_shape)

        for i, ID in enumerate(list_IDs_temp): 
            # print(i,end='\r')
            # ID is in format 'filenum-framenum'
            ID = ID.split('-')
            file_num = int(ID[0])
            if file_num > 9:
                continue
            frame_num = int(ID[1])+np.random.randint(0,N_JITTER-1)
            frame_num = np.minimum(frame_num,999-N_TIMESTEPS)
            frame_num = np.maximum(frame_num,0)

            # quicker way reading from hdf5
            file_name = folder_name + 'test_test1.hdf5' # 'hdf5_'+stub_name + str(file_num) + '.hdf5'
            h5file = h5py.File(file_name, 'r')

            for j in range(3, N_TIMESTEPS):
                try:
                    X[i,j] = h5file['frame_'+str(frame_num+j)+'_x'][:] # /255
                except:
                    continue
                y[i,j,:-2] = h5file['frame_'+str(frame_num+j)+'_y'][:]

                help_i = h5file['frame_'+str(frame_num+j)+'_helperarr'][:]
                kill_i = help_i[0]
                dead_i = help_i[1]
                shoot_i = y[i,j,n_keys:n_keys+1] # all these are binary variables
                reward_i = kill_i - 0.5*dead_i - 0.01*shoot_i # this is reward function
                y[i,j,-2:] = (reward_i,0.) # 0. is a placeholder for original advantage

                # for mouse, we're going to use a manual hack to remove most extreme 2 classes
                if y[i,j,n_keys+n_clicks] == 1:
                    y[i,j,n_keys+n_clicks] = 0
                    y[i,j,n_keys+n_clicks+2] = 1
                elif y[i,j,n_keys+n_clicks+1] == 1:
                    y[i,j,n_keys+n_clicks+1] = 0
                    y[i,j,n_keys+n_clicks+2] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x-1] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x-1] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x-3] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x-2] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x-2] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x-3] = 1

                # same for mouse y as of 20 aug
                if y[i,j,n_keys+n_clicks+n_mouse_x] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+2] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x+1] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x+1] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+2] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-1] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-1] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-3] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-2] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-2] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-3] = 1

            # add a manual hack here to make sure lclick is down  2 aug 2021
            # this is because firing rate of guns is slower than frame rate
            # yes I know should have done this at preprocessing stage...
            for j in range(1,N_TIMESTEPS-1):
                try:
                    if y[i,j-1,n_keys:n_keys+1] == 1 and y[i,j+1,n_keys:n_keys+1] == 1:
                        y[i,j,n_keys:n_keys+1] = 1
                except IndexError:
                    continue

            # 7 aug seem to need to fill in 1001 as well for spraying
            for j in range(1,N_TIMESTEPS-2):
                try:
                    if y[i,j-1,n_keys:n_keys+1] == 1 and y[i,j+2,n_keys:n_keys+1] == 1:
                        y[i,j,n_keys:n_keys+1] = 1
                        y[i,j+1,n_keys:n_keys+1] = 1
                except IndexError:
                    continue

            # TODO, include x_aux

            h5file.close()

            # do data aug
            # have the choice of mirroring image 
            # and accompanying mouse movement
            # this seemed to work ok for aim mode, but not deathmatch
            if IS_MIRROR:
                if np.random.rand()<0.3:
                    X[i] = np.flip(X[i],-2) # flip width dim
                    # also need to flip mouse x movement
                    y[i,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x] = np.flip(y[i,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x],axis=-1)
                    # also must flip 'a' and 'd' keys
                    akey = y[i,:,1]
                    dkey = y[i,:,3]
                    y[i,:,1] = dkey
                    y[i,:,3] = akey
            if i<len(X):
                # brightness
                if np.random.rand()<0.5: # was 0.2, raised to 0.5
                    # adjust in range 0.7 to 1.1, <1 darkesns, >1 brightens
                    bright = np.random.rand()*0.6+0.7
                    X[i] *= bright
                    X[i] = np.clip(X[i],0,255).astype(int)

                # contrast
                # follow https://stackoverflow.com/questions/49142561/change-contrast-in-numpy/49142934
                if np.random.rand()<0.5:
                    contrast = np.random.rand()*0.6+0.7
                    X[i] = np.clip(128 + contrast * X[i] - contrast * 128, 0, 255).astype(int)

        return X, y

N_JITTER = 20 # number frames to randomly offset by, going forward only!
# data_list = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list1 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list2 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list3 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list4 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list_full = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]


print('data_list1 training on sequences: ',len(data_list1))
print('data_list1 training on frames: ',len(data_list1*N_TIMESTEPS))

print('data_list2 training on sequences: ',len(data_list2))
print('data_list2 training on frames: ',len(data_list2*N_TIMESTEPS))

print('data_list3 training on sequences: ',len(data_list3))
print('data_list3 training on frames: ',len(data_list3*N_TIMESTEPS))

print('data_list4 training on sequences: ',len(data_list4))
print('data_list4 training on frames: ',len(data_list4*N_TIMESTEPS))


np.random.shuffle(data_list1) # shuffle it (in place)
partition1 = {}
partition1['train'] = data_list1[:int(len(data_list1)*1.)]
partition1['validation'] = data_list1[int(len(data_list1)*0.995):]

np.random.shuffle(data_list2) # shuffle it (in place)
partition2 = {}
partition2['train'] = data_list2[:int(len(data_list2)*1.)]
partition2['validation'] = data_list2[int(len(data_list2)*0.995):]

np.random.shuffle(data_list3) # shuffle it (in place)
partition3 = {}
partition3['train'] = data_list3[:int(len(data_list3)*1.)]
partition3['validation'] = data_list3[int(len(data_list3)*0.995):]

np.random.shuffle(data_list4) # shuffle it (in place)
partition4 = {}
partition4['train'] = data_list4[:int(len(data_list4)*1.)]
partition4['validation'] = data_list4[int(len(data_list4)*0.995):]

# this is not subsampled
partition_full = {}
partition_full['tmp'] = data_list_full[:int(batch_size*2)]
partition_full['train_full'] = data_list_full[:int(len(data_list_full)*1.)]
partition_full['validation_full'] = data_list_full[int(len(data_list_full)*0.995):]


training_generator1 = DataGenerator(list_IDs=partition1['train'], batch_size=batch_size, shuffle=True)
validation_generator1 = DataGenerator(list_IDs=partition1['validation'], batch_size=batch_size, shuffle=True)

training_generator2 = DataGenerator(list_IDs=partition2['train'], batch_size=batch_size, shuffle=True)
validation_generator2 = DataGenerator(list_IDs=partition2['validation'], batch_size=batch_size, shuffle=True)

training_generator3 = DataGenerator(list_IDs=partition3['train'], batch_size=batch_size, shuffle=True)
validation_generator3 = DataGenerator(list_IDs=partition3['validation'], batch_size=batch_size, shuffle=True)

training_generator4 = DataGenerator(list_IDs=partition4['train'], batch_size=batch_size, shuffle=True)
validation_generator4 = DataGenerator(list_IDs=partition4['validation'], batch_size=batch_size, shuffle=True)

tmp_generator = DataGenerator(list_IDs=partition_full['tmp'], batch_size=batch_size, shuffle=True)
training_generator_full = DataGenerator(list_IDs=partition_full['train_full'], batch_size=batch_size, shuffle=True)
validation_generator_full = DataGenerator(list_IDs=partition_full['validation_full'], batch_size=batch_size, shuffle=True)

print('starting to train...')

for iter_letter in ['a','b','c','d','e','f']:
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=False, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+iter_letter+'4')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=False, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+iter_letter+'8')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=False, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+iter_letter+'12')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=False, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+iter_letter+'16')

# save my models
tp_save_model(model, save_dir, model_name)

print('took',np.round(time.time()-start_time,1),' secs\n')