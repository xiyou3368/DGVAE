import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.log_lik = self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerOur(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        
        labels_sub = tf.reshape(labels, [-1])
        self.cost =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.log_lik =  self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        if True == True:
          self.a = 0.01 * np.ones((1 , FLAGS.hidden2)).astype(np.float32)
          self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),-1)).T)
          self.var2 = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/FLAGS.hidden2) ) ).T +
                                ( 1.0/(FLAGS.hidden2*FLAGS.hidden2) )*np.sum(1.0/self.a,-1) ).T  )
          ## the KL loss for the c
          latent_loss = 1 * (tf.reduce_sum(tf.div(tf.exp(model.z_log_std), self.var2), -1) + \
                             tf.reduce_sum(tf.multiply(tf.div((self.mu2 - model.z_mean), self.var2),
                                                  (self.mu2 - model.z_mean)), -1) - FLAGS.hidden2 + \
                             tf.reduce_sum(tf.log(self.var2), -1) - tf.reduce_sum(model.z_log_std, -1))
          self.kl =  0.5/num_nodes * tf.reduce_mean(latent_loss) 

        self.cost +=  1 * self.kl
        self.opt_op_recon = self.optimizer.minimize(self.log_lik)
        self.opt_op_kl = self.optimizer.minimize(self.kl)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        labels_sub = tf.reshape(labels, [-1])
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
