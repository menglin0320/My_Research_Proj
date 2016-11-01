import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from ops import *

np.random.seed(0)
tf.set_random_seed(0)

#TODO load the data **********
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
#change the code here
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
class Adversarial-Autoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100,model_name = 'adversarial',
                 df_dim = 32,learning_rate_d = 0.001,learning_rate_vae = 0.0001,
                  x_dim = 28, y_dim = 28,c_dim = 1,beta1 = 0.9,grad_clip = 5.0):

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name =model_name
        self.df_dim = df_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.beta1 = beta1
        self.grad_clip = grad_clip
        self.fake_label = np.array(self.batch_size*[10], dtype=np.int32) # label of fake batches
        self.fake_label_one_hot = self.to_one_hot(self.fake_label)        
        # tf Graph input
        self.learning_rate_d =learning_rate_d
        self.learning_rate_vae = learning_rate_vae
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.d_bn1 = batch_norm(batch_size, name=self.model_name+'_d_bn1')
        self.d_bn2 = batch_norm(batch_size, name=self.model_name+'_d_bn2')

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer

        self.t_vars = tf.trainable_variables()

        self.q_vars = [var for var in self.t_vars if ('_q_') in var.name]
        self.g_vars = [var for var in self.t_vars if ('_g_') in var.name]
        self.d_vars = [var for var in self.t_vars if (self.model_name+'_d_') in var.name]        
        self.both_vars = self.q_vars+self.g_vars
        
        # clip gradients
        balanced_loss = 1.0*g_loss + 10.0 * self.vae_loss
        d_opt_real_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss_real, self.d_vars), self.grad_clip)
        d_opt_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_vars), self.grad_clip)
        g_opt_grads, _ = tf.clip_by_global_norm(tf.gradients(self.balanced_loss, self.both_vars), self.grad_clip)
        vae_opt_grads, _ = tf.clip_by_global_norm(tf.gradients(self.vae_loss, self.q_vars), self.grad_clip)

        d_real_optimizer = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1)
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1)
        g_optimizer = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1)
        vae_optimizer = tf.train.AdamOptimizer(self.learning_rate_vae, beta1=self.beta1)

        self.d_opt_real = d_real_optimizer.apply_gradients(zip(d_opt_real_grads, self.d_vars))
        self.d_opt = d_optimizer.apply_gradients(zip(d_opt_grads, self.d_vars))
        self.g_opt = g_optimizer.apply_gradients(zip(g_opt_grads, self.both_vars))
        self.vae_opt = vae_optimizer.apply_gradients(zip(vae_opt_grads, self.q_vars))
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.all_variables())
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

        self.predict_real_samples = self.discriminator(tf.reshape(self.batch,\
                                    self.batch_size,self.x_dim,self.y_dim,self.c_dim)) # discriminiator on correct examples
        self.predict_fake_samples = self.discriminator(tf.reshape(self.x_reconstr_mean,\
                                    self.batch_size,self.x_dim,self.y_dim,self.c_dim), reuse=True) # feed generated images into D

        self.create_vae_loss_terms()
        self.create_gan_loss_terms()

    def partial_train(self, batch, label):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.

        I should really seperate the below tricks into parameters, like number of times/pass
        and also the regulator threshold levels.
        """

        counter = 0

        label_one_hot = self.to_one_hot(label)

        '''
        for i in range(1):
          counter += 1
          _, vae_loss = self.sess.run((self.vae_opt, self.vae_loss),
                                  feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
        '''

        for i in range(16):
          counter += 1
          _, g_loss, vae_loss, g_accuracy = self.sess.run((self.g_opt, self.g_loss, self.vae_loss, self.g_loss_accuracy),
                                  feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
          if g_accuracy > 0.98:
            break

        # train classifier on only real mnist digits
        # _ = self.sess.run((self.d_opt_real), feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})

        # calculate accuracy before deciding whether to train discriminator
        d_loss, d_loss_real, d_loss_fake, d_real_accuracy, d_fake_accuracy = self.sess.run((self.d_loss, self.d_loss_real, self.d_loss_fake, self.d_loss_real_accuracy, self.d_loss_fake_accuracy),
                                  feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})

        if d_fake_accuracy < 0.7 and g_accuracy > 0.6: # only train discriminiator if generator is good and d is behind.
          for i in range(8):
            counter += 1
            _, d_loss, d_loss_real, d_loss_fake, d_real_accuracy, d_fake_accuracy = self.sess.run((self.d_opt, self.d_loss, self.d_loss_real, self.d_loss_fake, self.d_loss_real_accuracy, self.d_loss_fake_accuracy),
                                    feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
            if d_fake_accuracy > 0.75:
              break
        elif d_real_accuracy < 0.6:
          for i in range(8):
            counter += 1
            _, d_real_accuracy = self.sess.run((self.d_opt_real, self.d_loss_real_accuracy), feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec, self.batch_label: label_one_hot})
            if d_real_accuracy > 0.7:
              break

    return d_loss, g_loss, vae_loss, counter, d_real_accuracy, d_fake_accuracy, g_accuracy, d_loss_real, d_loss_fake                
    
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        with tf.variable_scope("_q_"):
            all_weights['weights_recog'] = {
                'h1': tf.get_variable("weight", shape=[n_input, n_hidden_recog_1],
                initializer=tf.contrib.layers.xavier_initializer()),
                'h2': tf.get_variable("weight", shape=[n_hidden_recog_1, n_hidden_recog_2],
                initializer=tf.contrib.layers.xavier_initializer()),
                'out_mean':tf.get_variable("weight", shape=[n_hidden_recog_2, n_z],
                initializer=tf.contrib.layers.xavier_initializer())
                'out_log_sigma':tf.get_variable("weight", shape=[n_hidden_recog_2, n_z],
                initializer=tf.contrib.layers.xavier_initializer())
            all_weights['biases_recog'] = {
            tf.constant_initializer(0)
                'b1': tf.get_variable("weight", shape=[n_hidden_recog_1],
                initializer=tf.constant_initializer(0)),
                'b2': tf.get_variable("weight", shape=[n_hidden_recog_2],
                initializer=tf.constant_initializer(0)),
                'out_mean': tf.get_variable("weight", shape=[n_z],
                initializer=tf.constant_initializer(0)),
                'out_log_sigma': tf.get_variable("weight", shape=[n_z],
                initializer=tf.constant_initializer(0))}
        
        with tf.variable_scope("_g_"):
            all_weights['weights_gener'] = {
                'h1': tf.get_variable("weight", shape=[n_z, n_hidden_gener_1],
                initializer=tf.contrib.layers.xavier_initializer()),
                'h2': tf.get_variable("weight", shape=[n_hidden_gener_1, n_hidden_gener_2],
                initializer=tf.contrib.layers.xavier_initializer()),
                'out_mean':tf.get_variable("weight", shape=[n_hidden_gener_2, n_input],
                initializer=tf.contrib.layers.xavier_initializer())
                'out_log_sigma':tf.get_variable("weight", shape=[n_hidden_gener_2, n_input],
                initializer=tf.contrib.layers.xavier_initializer())}
            all_weights['biases_gener'] = {
                'b1': tf.get_variable("weight", shape=[n_hidden_gener_1],
                initializer=tf.constant_initializer(0)),
                'b2': tf.get_variable("weight", shape=[n_hidden_gener_2],
                initializer=tf.constant_initializer(0)),
                'out_mean': tf.get_variable("weight", shape=[n_input],
                initializer=tf.constant_initializer(0)),
                'out_log_sigma': tf.get_variable("weight", shape=[n_input],
                initializer=tf.constant_initializer(0))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
    def create_gan_loss_terms(self):            
        # Define loss function and optimiser
        ''' replace below with class-based disriminiator
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_right), self.D_right)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_wrong), self.D_wrong)
        self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(self.D_wrong), self.D_wrong)
        '''

        # cross entropy loss of predicting real mnist to real classes
        self.d_loss_real = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict_real_samples), reduction_indices=[1]))
        # accuracy of using discriminator as a normal mnist classifier
        self.d_loss_real_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_real_samples,1), tf.argmax(self.batch_label,1)), tf.float32))
        # cross entropy loss of predicting that fake generated mnist are in fact fake
        self.d_loss_fake = tf.reduce_mean(-tf.reduce_sum(self.fake_label_one_hot * tf.log(self.predict_fake_samples), reduction_indices=[1]))
        # accuracy of discriminator predicting a fake mnist digit
        self.d_loss_fake_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_fake_samples,1), tf.argmax(self.fake_label_one_hot,1)), tf.float32))
        # take the average of two d_loss to be the defacto d_loss
        self.d_loss = (10.0*self.d_loss_real + self.d_loss_fake)/ 11.0 # balanc out the classes
        # cross entropy of generator fooling discriminiator that its shit is real.
        self.g_loss = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict_fake_samples), reduction_indices=[1]))
        # accuracy of generated samples being fooled to be classified as their supposed ground truth labels
        self.g_loss_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_fake_samples,1), tf.argmax(self.batch_label,1)), tf.float32))        
    
    def create_vae_loss_terms(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.vae_loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})

    def create_gan_loss_terms(self):
        # Define loss function and optimiser
        ''' replace below with class-based disriminiator
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_right), self.D_right)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_wrong), self.D_wrong)
        self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(self.D_wrong), self.D_wrong)
        '''

        # cross entropy loss of predicting real mnist to real classes
        self.d_loss_real = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict_real_samples), reduction_indices=[1]))
        # accuracy of using discriminator as a normal mnist classifier
        self.d_loss_real_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_real_samples,1), tf.argmax(self.batch_label,1)), tf.float32))
        # cross entropy loss of predicting that fake generated mnist are in fact fake
        self.d_loss_fake = tf.reduce_mean(-tf.reduce_sum(self.fake_label_one_hot * tf.log(self.predict_fake_samples), reduction_indices=[1]))
        # accuracy of discriminator predicting a fake mnist digit
        self.d_loss_fake_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_fake_samples,1), tf.argmax(self.fake_label_one_hot,1)), tf.float32))
        # take the average of two d_loss to be the defacto d_loss
        self.d_loss = (10.0*self.d_loss_real + self.d_loss_fake)/ 11.0 # balanc out the classes
        # cross entropy of generator fooling discriminiator that its shit is real.
        self.g_loss = tf.reduce_mean(-tf.reduce_sum(self.batch_label * tf.log(self.predict_fake_samples), reduction_indices=[1]))
        # accuracy of generated samples being fooled to be classified as their supposed ground truth labels
        self.g_loss_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict_fake_samples,1), tf.argmax(self.batch_label,1)), tf.float32))


    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(image, self.df_dim, name=self.model_name+'_d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name=self.model_name+'_d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name=self.model_name+'_d_h2_conv')))
        h3 = linear(tf.reshape(h2, [self.batch_size, -1]), self.num_class, self.model_name+'_d_h2_lin')
        return tf.nn.softmax(h3)

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=75)
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize=(8, 12))
for i in range(5):

    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

vae_2d = train(network_architecture, training_epochs=75)

x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae_2d.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
plt.colorbar()


nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]])
        x_mean = vae_2d.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper")
plt.tight_layout()    

