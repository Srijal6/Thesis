import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def load_training(root_path, dir, batch_size, kwargs):
    transform = tf.contrib.image.transform(
        [tf.image.resize_images([256, 256]),
         tf.image.random_crop(224),
         tf.image.random_flip_left_right(),
         tf.convert_to_tensor()])
    data = mnist.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = tf.data.Dataset.soarce(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = tf.contrib.image.transform(
        [tf.image.resize_images([224, 224]),
         tf.convert_to_tensor()])
    data = mnist.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = tf.data.Dataset.soarce(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
