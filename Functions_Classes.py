import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.layers import *
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import Model
from  IPython.display import clear_output
import seaborn as sns
#for Kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
import tensorflow_datasets as tfds
from IPython.display import clear_output


#function to split datasets to train and test
def test_train_df_spliter(df,test_ratio = 0.2):
    train_df, test_df = train_test_split(df, test_size=test_ratio)
    return train_df.reset_index(drop = True) , test_df.reset_index(drop = True)

#================================================================================

#function to generate pca compponents
def pca_generator(df,num_pca = 3):
    np_df = df.values
    pca = PCA(n_components = num_pca)
    pca_values = pca.fit_transform(np_df)
    print('variance ratio is ',pca.explained_variance_ratio_)
    
    return pca_values

#================================================================================

#function to deal with outliers
def IDoutliers(pca_df,ppg_df ,ecg_df,threshold_list):
    i = 0
    for col in pca_df.columns:
        pca_df = pca_df.reset_index(drop = True)
        ppg_df = ppg_df.reset_index(drop = True)
        ecg_df = ecg_df.reset_index(drop = True)
        outlier_indexes = np.where((pca_df[col].values > threshold_list[i][0]) | (pca_df[col].values < threshold_list[i][1]))[0]
        print("outliers from ",col , " column is ",outlier_indexes.shape)
        ppg_df.drop(outlier_indexes,axis = 0,inplace = True)
        ecg_df.drop(outlier_indexes,axis = 0,inplace = True)
        pca_df.drop(outlier_indexes,axis = 0,inplace = True)
        i = i+1
    return pca_df , ppg_df , ecg_df  #dataframe without outliers

#================================================================================

#All together from PCA to cleaning
def all_together(ppg_df,ecg_df ,reference = 1 , z_threshold = [(2,-2),(1.5,-1),(1,-1.5)] , num_pca = 3 ):
    if reference == 1:
        pca_df_with = pd.DataFrame(pca_generator(ppg_df,num_pca), columns = ['pca1','pca2','pca3'])
    else:
         pca_df_with = pd.DataFrame(pca_generator(ecg_df,num_pca), columns = ['pca1','pca2','pca3'])
            
    pca_df_without , ppg_df_without , ecg_df_without = IDoutliers(pca_df_with,ppg_df,ecg_df,z_threshold)
    
    print("histogram of pca with and without outliers")
    print("shape with outliers is ",ppg_df.shape)
    print("shape without outliers is",ppg_df_without.shape)
    for i in range(3):
        
        plt.figure()
        plt.title('with outliers pca'+str(i+1))
        sns.histplot(pca_df_with.values[:,i])
        
        plt.figure()
        plt.title('without outliers pca'+str(i+1))
        sns.histplot(pca_df_without.values[:,i])
        
    return pca_df_without ,pca_df_with, ppg_df_without , ecg_df_without

#================================================================================

#Clustering
def clusterer(dataframe , num_clusters = 3):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(dataframe)
    cluster_labels = kmeans.labels_  
    return cluster_labels

#================================================================================

#Plotter function
def plotter(dataframe , labels):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax = plt.axes(projection='3d')
    """ax.set_xlim3d(-26, 27)
    ax.set_ylim3d(-17, 25)
    ax.set_zlim3d(-17, 10)"""
    # Data for a three-dimensional line
    cols = dataframe.columns
    zdata = dataframe[cols[0]].values
    xdata = dataframe[cols[1]].values
    ydata = dataframe[cols[2]].values
    #index = np.random.randint(0,zdata.shape[0],X.shape[0])
    ax.scatter3D(xdata, ydata, zdata, c=labels ,cmap = 'gist_rainbow')

#================================================================================

#To write each cluster in CSV file
def classes_csv_writer(df , class_labels,name):
    unique_classes = np.unique(class_labels)
    print("classes are ",unique_classes)
    for i in unique_classes :
        path = "/content/gdrive/MyDrive/most cleaned version beats/"+name + "_class_"+str(i)+".csv"
        df.iloc[class_labels == i,:].to_csv(path)
      
        print("class ",i," was written")
    print("Writting complete")

#================================================================================
#function to generate dataset in form tensor and batches

def get_dataset(training_df,is_validation=False):
    '''Loads and prepares the mnist dataset from TFDS'''
    if is_validation:
        split_name = "test"
    else:
        split_name = "train"
    tf.executing_eagerly()
    features = list(map(int,list(range(0,training_df.shape[1]))))
    training_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(training_df[features].values, tf.float32),)))
    if is_validation:
        dataset = training_dataset.batch(BATCH_SIZE)
    else:
        dataset = training_dataset.shuffle(120).batch(BATCH_SIZE)
    return dataset

#================================================================================
#Variational autoencoder functions
#Sampling layer Class

class Sampling(tf.keras.layers.Layer):
    
    def call(self, inputs):
        """Generates a random sample and combines with encoder output

        Args:
          inputs -- output tensor from the encoder

        Returns:
      `inputs` tensors combined with a random sample
        """

        # unpack the output of the encoder
        mu, sigma = inputs

        # get the size and dimensions of the batch
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        # generate a random tensor
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # combine the inputs and noise
        return mu + tf.exp(0.5 * sigma) * epsilon
    def get_config(self):
      config = super(Sampling, self).get_config()
      return config

#***************************************************************************
#layers of encoder part definition

def encoder_layers(inputs, latent_dim):
  """Defines the encoder's layers
  Args:
    inputs -- batch from the dataset
    latent_dim -- dimensionality of the latent space
  
  Returns:
    mu -- learned mean
    sigma -- learned standard deviation
    batch_2.shape -- shape of the features before flattening
  """

  x = Conv1D(256, 5, padding="same")(inputs)
  x = LeakyReLU()(x)
  x = BatchNormalization()(x)
  x = MaxPooling1D(2, padding='same')(x)
    
  x = Conv1D(512, 5, padding="same")(inputs)
  x = LeakyReLU()(x)
  x = BatchNormalization()(x)
  x = MaxPooling1D(2, padding='same')(x)

  x = Conv1D(128, 5, padding='same')(x)
  x = LeakyReLU()(x)
  x = BatchNormalization()(x)
  x = MaxPooling1D(2, padding='same')(x)

  x = Conv1D(64, 3, padding='same')(x)
  x = LeakyReLU()(x)
  x = BatchNormalization()(x)
  x = MaxPooling1D(2, padding='same')(x)

  x = Conv1D(32,3, activation='relu', padding='same')(x)
  batch_2 = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Flatten(name="encode_flatten")(batch_2)

  

  # we arbitrarily used 20 units here but feel free to change and see what results you get
  x = tf.keras.layers.Dense(60, activation="relu", name="encoder_dense")(x)
  x = tf.keras.layers.BatchNormalization()(x)

  # add output Dense networks for mu and sigma, units equal to the declared latent_dim
  mu = tf.keras.layers.Dense(latent_dim, name="latent_mu")(x)
  sigma = tf.keras.layers.Dense(latent_dim, name="latent_sigma")(x)

  return mu, sigma, batch_2.shape

#================================================================================
def encoder_model(latent_dim, input_shape):
    """Defines the encoder model with the Sampling layer

     Args:
      latent_dim -- dimensionality of the latent space
      input_shape -- shape of the dataset batch

    Returns:
     model -- the encoder model
      conv_shape -- shape of the features before flattening
    """

    # declare the inputs tensor with the given shape
    inputs = tf.keras.layers.Input(shape=input_shape)

    # get the output of the encoder_layers() function
    mu, sigma, conv_shape = encoder_layers(inputs, latent_dim=LATENT_DIM)

    # feed mu and sigma to the sampling layer
    z = Sampling()((mu, sigma))

    # build the whole encoder model
    model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma, z])

    return model, conv_shape

#******************************************************************************
#decoder layers
def decoder_layers(inputs, conv_shape):
  """Defines the decoder layer
  Args:
    inputs -- output of the decoder
    conv_shape -- shape of the features before flattening

  Returns:
    tensor containing the decoded output
  """

  # feed to a Dense network with units computed from the conv_shape dimensions
  units = conv_shape[1] * conv_shape[2] 
  x = tf.keras.layers.Dense(units, name="decode_dense1")(inputs)
  x = LeakyReLU()(x)
  x = tf.keras.layers.BatchNormalization()(x)

  # reshape output using the conv_shape dimensions
  x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2]), name="decode_reshape")(x)

  # upsample the features back to the original dimensions
  x = Conv1DTranspose(32, 3 , padding='same')(x)
  x = LeakyReLU()(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = UpSampling1D(2)(x)
    
  x = Conv1DTranspose(64, 3, padding='same')(x)
  x = LeakyReLU()(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = UpSampling1D(2)(x)
    
  x = Conv1DTranspose(128, 5, padding='same')(x)
  x = LeakyReLU()(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = UpSampling1D(2)(x)
  
  x = Conv1DTranspose(512, 5, activation='relu', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  #x = UpSampling1D(2)(x)
    
  
  x = Conv1DTranspose(1, 5, activation='relu', padding='same')(x)
 # x = UpSampling1D(2)(x)
  #x = Dense(120, activation='relu')(x)

  return x
#========================================================================
#defining decoder model

def decoder_model(latent_dim, conv_shape):
    inputs = tf.keras.layers.Input(shape=(latent_dim,))

    # get the output of the decoder layers
    outputs = decoder_layers(inputs, conv_shape)

    # declare the inputs and outputs of the model
    model = tf.keras.Model(inputs, outputs)

    return model

#*****************************************************************************
#loss function of variational autoencoder

def kl_reconstruction_loss(inputs, outputs, mu, sigma):
    
    # honestly, to truly understand this loss is challenging to me
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * - 0.5

    return kl_loss

#=================================================================================
#define VAE model function
 
def vae_model(encoder, decoder, input_shape):

    # set the inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # get mu, sigma, and z from the encoder output
    mu, sigma, z = encoder(inputs)

    # get reconstructed output from the encoder
    reconstructed = decoder(z)

    # define the inputs and outputs to the VAE
    model = tf.keras.Model(inputs=inputs, outputs=reconstructed)

    # add the KL loss
    loss = kl_reconstruction_loss(inputs, z, mu, sigma)
    model.add_loss(loss)

    return model

#=====================================================================================
#all together

def get_models(input_shape, latent_dim):
  
    encoder, conv_shape = encoder_model(latent_dim=latent_dim, input_shape=input_shape)
    decoder = decoder_model(latent_dim=latent_dim, conv_shape=conv_shape)
    vae = vae_model(encoder, decoder, input_shape=input_shape)
    return encoder, decoder, vae

#======================================================================================
#indicator of model performance functions
def generate_and_save_images(model, epoch, step, test_input):
    # generate images from the test input
    predictions = model.predict(test_input)

    # plot the results
    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.plot(predictions[i, :, 0])
      

    # tight_layout minimizes the overlap between 2 sub-plots
    fig.suptitle("epoch: {}, step: {}".format(epoch, step))
    plt.savefig('image_at_epoch_{:04d}_step{:04d}.png'.format(epoch, step))
    plt.show()
#-------------------------------------------------------------------------------------
def modified_generate_and_save_images(model, epoch, step, test_input,samples):
  """Helper function to plot our 16 images

  Args:

  model -- the decoder model
  epoch -- current epoch number during training
  step -- current step number during training
  test_input -- random tensor with shape (16, LATENT_DIM)
  """

  # generate images from the test input
  samples = np.random.randint(0 , 3399,8)
  predictions = model.predict(test_input[samples])
  y = y_test[samples]
  
  # plot the results
  fig = plt.figure(figsize=(15,10))

  for i in range(8):
      plt.subplot(2, 4, i+1)
      plt.plot(predictions[i, :, 0],'b--')
      plt.plot(y[i,:,0],'r')
      #plt.plot(test_input[samples][i,:,0],'g')

  # tight_layout minimizes the overlap between 2 sub-plots
  fig.suptitle("epoch: {}, step: {}".format(epoch, step))
  #plt.savefig('image_at_epoch_{:04d}_step{:04d}.png'.format(epoch, step))
  plt.show()






