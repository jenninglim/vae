import tensorflow as tf
from utils import batch_index_groups, dtype

class vae(object):
    def __init__(self, y_input, dim_img, dim_z, num_hidden):
        
        self.n_hidden = num_hidden
        self.keep_prob = 0.9
        learn_rate = 1e-3

        mean, stddev = self.encoder(y_input, dim_z)

        samples = mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype = dtype)
        epsilon = 1e-6
        y_output = tf.clip_by_value(t=self.decoder(samples, dim_img),
                        clip_value_min=epsilon,
                        clip_value_max=1 - epsilon
                        )
        neg_log_likeli, kl = self.loss(mean, stddev, y_output, y_input)
        self.neg_marginal_likelihood =neg_log_likeli
        self.kl_divergence = kl
        self.loss = kl + neg_log_likeli

        
    # Gaussian MLP
    def encoder(self, input_image, n_output):
        with tf.name_scope("encoder"):
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0., dtype=dtype)

            # First Layer
            w0 = tf.get_variable("w0", [input_image.get_shape()[1], self.n_hidden], initializer=w_init, dtype=dtype)
            b0 = tf.get_variable("b0", [self.n_hidden], initializer=b_init, dtype=dtype)

            h0 = tf.nn.dropout(tf.nn.elu(tf.matmul(input_image, w0) + b0), self.keep_prob)
           
           # Second Layer
            w1 = tf.get_variable("w1", [h0.get_shape()[1], self.n_hidden], initializer=w_init, dtype=dtype)
            b1 = tf.get_variable("b1", [self.n_hidden], initializer=b_init, dtype=dtype)

            h1 = tf.nn.dropout(tf.nn.elu(tf.matmul(h0, w1) + b1), self.keep_prob)

            # Output Layer

            w2 = tf.get_variable("w2", [h1.get_shape()[1], n_output * 2], initializer=w_init, dtype=dtype)
            b2 = tf.get_variable("b2", [n_output * 2], initializer=b_init, dtype=dtype)

            gauss = tf.matmul(h1, w2) + b2
            
            mean = gauss[:, :n_output]

            # stddev must be > 0
            stddev = 1e-6  + tf.nn.softplus(gauss[:, n_output:])
        return mean, stddev

    # Bernoulli MLP
    def decoder(self, z, n_output):
        with tf.name_scope("decoder"):
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0., dtype=dtype)

            # First Layer
            w0_d = tf.get_variable("w0_d", [z.get_shape()[1], self.n_hidden], initializer=w_init, dtype=dtype)
            b0_d = tf.get_variable("b0_d", [self.n_hidden], initializer=b_init, dtype=dtype)

            h0 = tf.nn.dropout(tf.nn.elu(tf.matmul(z, w0_d) + b0_d), self.keep_prob)
           
           # Second Layer
            w1_d = tf.get_variable("w1_d", [h0.get_shape()[1], self.n_hidden], initializer=w_init, dtype=dtype)
            b1_d = tf.get_variable("b1_d", [self.n_hidden], initializer=b_init, dtype=dtype)

            h1 = tf.nn.dropout(tf.nn.elu(tf.matmul(h0, w1_d) + b1_d), self.keep_prob)

            # Output Layer

            w2_d = tf.get_variable("w2_d", [h1.get_shape()[1], n_output], initializer=w_init, dtype=dtype)
            b2_d = tf.get_variable("b2_d", [n_output], initializer=b_init, dtype=dtype)
           
            y = tf.sigmoid(tf.matmul(h1, w2_d) + b2_d)
        return y

    def loss(self, mean, stddev, y_output, y_output_true):
        # bernoulli log likelihood
        log_likeli = tf.reduce_mean(tf.reduce_sum(y_output_true * tf.log(y_output) + (1 - y_output_true) * tf.log(1-y_output), axis = 1))
        
    
        # Kl
        std_squared = tf.square(stddev)
        mea_squared = tf.square(mean)
        kl = tf.reduce_mean(1/2 * tf.reduce_sum( tf.log(std_squared) + mea_squared + std_squared - 1, axis = 1))
   
        return - log_likeli, kl
