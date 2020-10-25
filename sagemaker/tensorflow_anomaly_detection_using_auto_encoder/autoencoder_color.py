import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import os
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Autoencoder(Model):
  def __init__(self, size):
    super(Autoencoder, self).__init__()

    input_shape = (size, size, 3)

    self.encoder = tf.keras.Sequential([
         # 28 x 28 x 1
        layers.Conv2D(32, kernel_size=(3, 3),
                                activation='relu', padding='same',
                                input_shape=input_shape),
        # 28 x 28 x 16
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        # 14 x 14 x 16
        layers.Conv2D(32, kernel_size=(3, 3),
                                activation='relu', padding='same'),
        # 14 x 14 x 8
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        # 7 x 7 x 8
        layers.Conv2D(64, kernel_size=(3, 3),
                            activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same')
    ])

    self.encoder.summary()

    self.decoder = tf.keras.Sequential([
        # 7 x 7x 8
        layers.Conv2D(64, kernel_size=(3, 3),
                                activation='relu', padding='same'),
        # 7 x 7 x 8
        layers.UpSampling2D(size=(2, 2)),
        # 14 x 14 x 8
        layers.Conv2D(32, kernel_size=(3, 3),
                                activation='relu', padding='same'),
        # 14 x 14 x 32
        layers.UpSampling2D(size=(2, 2)),
        # 28 x 28 x 32
        layers.Conv2D(32, kernel_size=(3, 3),
                                activation='relu', padding='same'),
        # 28 x 28 x 32
        layers.UpSampling2D(size=(2, 2)),
        # 28 x 28 x 16
        layers.Conv2D(3, kernel_size=(3, 3),
                                activation='sigmoid', padding='same')
    ])
#     self.decoder.summary()



  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
def _load_data(base_dir, filename):
    """Load data"""
    data = np.load(os.path.join(base_dir, filename))
    return data


# def _load_testing_data(base_dir):
#     """Load MNIST testing data"""
#     x_test = np.load(os.path.join(base_dir, 'x_test_smp.npy'))
#     return x_test
    
def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--train_data_name', type=str, default='')
    parser.add_argument('--valid_data_name', type=str, default='')

    return parser.parse_known_args()

    
if __name__ == "__main__":
    args, unknown = _parse_args()
    
#     print(os.listdir(args.train))

    train_data = _load_data(args.train, args.train_data_name)
    eval_data = _load_data(args.train, args.valid_data_name)
    
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = 64 * strategy.num_replicas_in_sync

    with strategy.scope():
        
        autoencoder = Autoencoder(args.size) 

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.model_dir, histogram_freq=1)

    autoencoder.fit(train_data, train_data,
                epochs=args.epochs,
                shuffle=True,
                validation_data=(eval_data, eval_data),
#                 callbacks=[tensorboard_callback],
                 batch_size=BATCH_SIZE)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        autoencoder.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
        