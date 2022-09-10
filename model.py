import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, ReLU, Add, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np

def build_model():
    c=1e-4
    inputs=Input((2,8,8))
    # 3 layers of convolution
    conv1=Conv2D(filters=16, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(inputs)
    #conv1.trainable=False
    conv1=BatchNormalization(axis=[1,2])(conv1)
    conv1=ReLU()(conv1)
    conv2=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(conv1)
    #conv2.trainable=False
    conv2=BatchNormalization(axis=[1,2])(conv2)
    conv2=ReLU()(conv2)
    conv3=Conv2D(filters=64, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(conv2)
    conv3=BatchNormalization(axis=[1,2])(conv3)
    convolution=ReLU()(conv3)
    #shortcut path
    #convolution=Add()([conv1,conv3])
    #convolution=ReLU()(convolution)
    #policy head
    policy1=Conv2D(filters=2, kernel_size=(1,1),
                   data_format="channels_first", activation="relu",
                   kernel_regularizer=l2(c))(convolution)
    policy2=Flatten()(policy1)
    policy_output=Dense(64, activation="softmax",
                        kernel_regularizer=l2(c))(policy2)
    #value head 
    value1=Conv2D(filters=1, kernel_size=(1,1),
                  data_format="channels_first", activation="relu",
                  kernel_regularizer=l2(c))(convolution)
    value2=Flatten()(value1)
    value3=Dense(64, activation="relu",
                 kernel_regularizer=l2(c))(value2)
    value_output=Dense(1, activation="tanh",
                       kernel_regularizer=l2(c))(value3)
    model=Model(inputs, [policy_output,value_output])
    return model

old_model=tf.keras.models.load_model("checkpoint2-selfplay-bn.h5")
weights=old_model.get_weights()

new_model=tf.keras.models.load_model("checkpoint2-policy-round1.pickle.h5")
new_weights=new_model.get_weights()
for i in range(len(weights)):
    print(i+1)
    print(len(weights[i]))
    print(np.array_equal(weights[i],new_weights[i]))
    print()
    
'''
model=build_model()
model.set_weights(weights)

conv_layers=["conv2d","batch_normalization","conv2d_1","batch_normalization_1",
                 "conv2d_2","batch_normalization_2"]
policy_layers=["conv2d_3","dense"]
value_layers=["conv2d_4","dense_1","dense_2"]
for layer in conv_layers+value_layers:
    model.get_layer(layer).trainable=False
print(model.summary())

optimizer = Adam()
losses = ['categorical_crossentropy', 'mean_squared_error'] # for p and v
model.compile(optimizer=optimizer, loss=losses, loss_weights=[1,1])


model.save("checkpoint2-policy-bn.h5")
'''
#------------------------------------------------------------------------------
def build_model_mentor():
    c=1e-4
    inputs=Input((2,8,8))
    # 3 layers of convolution
    conv1=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(inputs)
    conv2=Conv2D(filters=64, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(conv1)
    conv3=Conv2D(filters=128, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(conv2)
    '''
    #shortcut path
    shortcut=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(inputs)
    convolution=Add()([conv3,shortcut])
    convolution=ReLU()(convolution)
    '''
    #policy head
    policy1=Conv2D(filters=4, kernel_size=(1,1),
                   data_format="channels_first", activation="relu",
                   kernel_regularizer=l2(c))(conv3)
    policy2=Flatten()(policy1)
    policy_output=Dense(64, activation="softmax",
                        kernel_regularizer=l2(c))(policy2)
    #value head 
    value1=Conv2D(filters=2, kernel_size=(1,1),
                  data_format="channels_first", activation="relu",
                  kernel_regularizer=l2(c))(conv3)
    value2=Flatten()(value1)
    value3=Dense(64, activation="relu",
                 kernel_regularizer=l2(c))(value2)
    value_output=Dense(1, activation="tanh",
                       kernel_regularizer=l2(c))(value3)
    model=Model(inputs, [policy_output,value_output])
    return model

def build_model_1_3():
    c=1e-4
    inputs=Input((2,8,8))
    # 3 layers of convolution
    conv1=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(inputs)
    conv2=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(conv1)
    conv3=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(conv2)
    #shortcut path
    shortcut=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(inputs)
    convolution=Add()([conv3,shortcut])
    convolution=ReLU()(convolution)
    #policy head
    policy1=Conv2D(filters=2, kernel_size=(1,1),
                   data_format="channels_first", activation="relu",
                   kernel_regularizer=l2(c))(convolution)
    policy2=Flatten()(policy1)
    policy_output=Dense(64, activation="softmax",
                        kernel_regularizer=l2(c))(policy2)
    #value head 
    value1=Conv2D(filters=2, kernel_size=(1,1),
                  data_format="channels_first", activation="relu",
                  kernel_regularizer=l2(c))(convolution)
    value2=Flatten()(value1)
    value3=Dense(64, activation="relu",
                 kernel_regularizer=l2(c))(value2)
    value_output=Dense(1, activation="tanh",
                       kernel_regularizer=l2(c))(value3)
    model=Model(inputs, [policy_output,value_output])
    return model

def build_model_new():
    c=1e-4
    inputs=Input((2,8,8))
    # 3 layers of convolution
    conv1=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(inputs)
    conv2=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(conv1)
    conv3=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first", activation="relu",
                 kernel_regularizer=l2(c))(conv2)
    #shortcut path
    convolution=Add()([conv1,conv3])
    convolution=ReLU()(convolution)
    #policy head
    policy1=Conv2D(filters=2, kernel_size=(1,1),
                   data_format="channels_first", activation="relu",
                   kernel_regularizer=l2(c))(convolution)
    policy2=Flatten()(policy1)
    policy_output=Dense(64, activation="softmax",
                        kernel_regularizer=l2(c))(policy2)
    #value head 
    value1=Conv2D(filters=1, kernel_size=(1,1),
                  data_format="channels_first", activation="relu",
                  kernel_regularizer=l2(c))(convolution)
    value2=Flatten()(value1)
    value3=Dense(64, activation="relu",
                 kernel_regularizer=l2(c))(value2)
    value_output=Dense(1, activation="tanh",
                       kernel_regularizer=l2(c))(value3)
    model=Model(inputs, [policy_output,value_output])
    return model

def build_model_bn():
    c=1e-4
    inputs=Input((2,8,8))
    # 3 layers of convolution
    conv1=Conv2D(filters=16, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(inputs)
    conv1=BatchNormalization(axis=[1,2])(conv1)
    conv1=ReLU()(conv1)
    
    conv2=Conv2D(filters=32, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(conv1)
    conv2=BatchNormalization(axis=[1,2])(conv2)
    conv2=ReLU()(conv2)
    conv3=Conv2D(filters=64, kernel_size=(3,3), padding="same",
                 data_format="channels_first",
                 kernel_regularizer=l2(c))(conv2)
    conv3=BatchNormalization(axis=[1,2])(conv3)
    convolution=ReLU()(conv3)
    #shortcut path
    #convolution=Add()([conv1,conv3])
    #convolution=ReLU()(convolution)
    #policy head
    policy1=Conv2D(filters=2, kernel_size=(1,1),
                   data_format="channels_first", activation="relu",
                   kernel_regularizer=l2(c))(convolution)
    policy2=Flatten()(policy1)
    policy_output=Dense(64, activation="softmax",
                        kernel_regularizer=l2(c))(policy2)
    #value head 
    value1=Conv2D(filters=1, kernel_size=(1,1),
                  data_format="channels_first", activation="relu",
                  kernel_regularizer=l2(c))(convolution)
    value2=Flatten()(value1)
    value3=Dense(64, activation="relu",
                 kernel_regularizer=l2(c))(value2)
    value_output=Dense(1, activation="tanh",
                       kernel_regularizer=l2(c))(value3)
    model=Model(inputs, [policy_output,value_output])
    return model

