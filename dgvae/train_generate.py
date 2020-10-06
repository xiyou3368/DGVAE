from __future__ import division
from __future__ import print_function
import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import sys
sys.path.append(os.path.abspath(r"."))
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from dgvae.optimizer import OptimizerAE, OptimizerVAE, OptimizerOur
from dgvae.optimizer import OptimizerVAE as Optimizer_graphite
from dgvae.input_data import load_data, generate_data
from dgvae.model import GCNModelAE, GCNModelVAE, OurModelAE, OurModelVAE
from dgvae.model_graphite import GCNModelFeedback
from dgvae.preprocessing import preprocess_graph,preprocess_graph_e,construct_feed_dict, sparse_to_tuple, mask_test_edges
from dgvae.preprocessing import mask_test_graphs, graph_padding,preprocess_graph_generate, preprocess_graph_generate_e

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 4.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('vae', 1, '1 for variational objective')  ## here is for the graphite

flags.DEFINE_string('model', 'our_vae', 'Model string.') #our_ae,our_vae,gcn_ae,gcn_vae,graphite_ae,graphite_vae
flags.DEFINE_string('dataset', 'Erdos_Renyi', 'Dataset string.') #Erdos_Renyi,Ego,Regular,Geometric,Power_Law,Barabasi_Albert
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('num_graph', 300, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('graph_max_size', 20, 'Whether to use features (1) or not (0).')
## the flags for graphite
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphite')
model_str = FLAGS.model
dataset_str = FLAGS.dataset
num_graph = FLAGS.num_graph
graph_max_size = FLAGS.graph_max_size
seed = 133
np.random.seed(seed)
tf.set_random_seed(seed)
graph_list, graph_size = generate_data(dataset_str, num_graph, [graph_max_size, graph_max_size], seed = seed)
graph_list, max_size = graph_padding(graph_list, graph_size)
features = sp.identity(int(max_size))

## split the train and valid and test set
graph_train, train_size,graph_val,val_size, graph_test, test_size = mask_test_graphs(graph_list, graph_size)
if model_str.startswith("our"):
  adj_norms, adj_labels = preprocess_graph_generate_e(graph_train)
  adj_norms_val, adj_labels_val = preprocess_graph_generate_e(graph_val)
  adj_norms_test, adj_labels_test = preprocess_graph_generate_e(graph_test)
else:
  adj_norms, adj_labels = preprocess_graph_generate(graph_train)
  adj_norms_val, adj_labels_val = preprocess_graph_generate(graph_val)
  adj_norms_test, adj_labels_test = preprocess_graph_generate(graph_test)
# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = max_size
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'our_ae':
    model = OurModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'our_vae':
    model = OurModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == "graphite_ae":
    FLAGS.vae = 0
    model = GCNModelFeedback(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == "graphite_vae":
    FLAGS.vae = 1
    model = GCNModelFeedback(placeholders, num_features, num_nodes, features_nonzero)
adj = graph_list[0]
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'our_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)
    elif model_str == 'graphite_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str ==  "graphite_vae":
        opt = Optimizer_graphite(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)
    elif model_str == 'our_vae':
        opt = OptimizerOur(preds=model.reconstructions,
                           labels=tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

def get_roc_score(adj_norm, adj_label, adj_orig,size, emb=None):
    """
    get the ROC score ,AP score, reconstruction error and neg log-likelyhood
    adj_norm:  the normalized adjs to calculated. It is a tuple for feed_dict.
    adj_label: the A+I to calculate. It is a tuple for feed_dict.
    adj_orig: the A+I as label of the adj.
    size: the size of the graphs. size <= max_size
    """
    size = int(size)
    if emb is None:
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return .5 * (1 + np.tanh(.5 * x))

    # Predict on test set of edges
    if model_str.startswith('gcn'):
        adj_rec = np.dot(emb, emb.T)
    if model_str.startswith("graphite"):
        adj_rec = sess.run(model.reconstructions_noiseless, feed_dict=feed_dict)
        adj_rec = adj_rec.reshape(adj_orig.shape)
    if model_str.startswith("our"):
        adj_rec = np.dot(emb, emb.T)
    preds_all = adj_rec[:size, :size]
    preds_all = preds_all.flatten()
    preds_all = sigmoid(preds_all)
    labels_all = adj_orig[:size, :size]
    labels_all = labels_all.flatten()
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    ## reconstruction err
    recons_error = np.mean(np.abs(labels_all - preds_all))
    ## log_like
    neg_log_like, cost = sess.run([opt.log_lik, opt.cost], feed_dict = feed_dict)
    ## the RMSE
    rmse = np.sqrt(np.mean(np.square(labels_all - preds_all)))
    return roc_score, ap_score, recons_error,neg_log_like, rmse

cost_val = []
acc_val = []
val_roc_score = []

if model_str.startswith("our_vae"):
  FLAGS.epochs = int(FLAGS.epochs/2)

# Train model
for epoch in range(FLAGS.epochs):
    ## enumerate the graphs
    for idx in range(len(adj_norms)):
        ## construct the adj_norm and adj_label
        adj_norm = adj_norms[idx]
        adj_label = adj_labels[idx]
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        if model_str.startswith("our_vae"):
          # Run the construction_part
          sub_iter = 2
          sub_loss = 0
          sub_loss_num = 0
          sub_pre_loss = 1e4
          outs = 0
          for i in range(sub_iter):
            outs = sess.run([opt.opt_op_recon, opt.log_lik, opt.accuracy], feed_dict = feed_dict)
            sub_loss += outs[1]
            sub_loss_num += 1
            if sub_iter % 15 == 0:
                sub_loss_mean = sub_loss / sub_loss_num
                if sub_pre_loss - sub_loss_mean< 1e-2:
                    sub_pre_loss = sub_loss_mean
                    print(sub_iter)
                    break
                sub_pre_loss = sub_loss_mean
                sub_loss = sub_loss_num = 0
        ##Run the ELBO
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
        #outs = sess.run([opt.opt_op, opt.cost,opt.log_lik,opt.kl, opt.accuracy], feed_dict=feed_dict)
    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    ## the test process on valid set
    val_roc_score_temp = []
    ap_curr_temp = []
    recon_error_temp = []
    log_like_temp = []
    for idx, adj in enumerate(graph_val):
        adj_orig = adj + np.identity(adj.shape[0])
        roc_curr, ap_curr, recon_error, log_like, rmse= get_roc_score(adj_norms_val[idx], adj_labels_val[idx],adj_orig, val_size[idx])
        val_roc_score_temp.append(roc_curr)
        ap_curr_temp.append(ap_curr)
        recon_error_temp.append(recon_error)
        log_like_temp.append(log_like)
    val_roc_score.append(np.mean(val_roc_score_temp))
    ap_curr = np.mean(ap_curr_temp)
    recon_err = np.mean(recon_error_temp)
    log_like = np.mean(log_like_temp)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "reconstruction error=", "{:.5f}".format(recon_err),
          "log_like=", "{:.5f}".format(log_like),
          "RMSE=", "{:.5f}".format(rmse),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
roc_test = []
ap_test = []
recon_err = []
log_like = []
for idx, adj in enumerate(graph_test):
    adj_orig = adj + np.identity(adj.shape[0])
    roc_score_temp, ap_score_temp, recon_err_temp, log_like_temp, rmse = get_roc_score(adj_norms_test[idx],adj_labels_test[idx],adj_orig, test_size[idx])
    roc_test.append(roc_score_temp)
    ap_test.append(ap_score_temp)
    recon_err.append(recon_err_temp)
    log_like.append(log_like_temp)
print('Test ROC score: ' + str(np.mean(roc_test)))
print('Test AP score: ' + str(np.mean(ap_test)))
print("Test Reconstruction error:" + str(np.mean(recon_err)))
print("Test Log Likelihood:" + str(np.mean(log_like)))
print("Test RMSE:" + str(np.mean(rmse)))
