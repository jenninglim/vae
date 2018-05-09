import tensorflow as tf
import os
import os.path
import numpy as np

from utils import batch_index_groups, dtype
import input_data
from vae import vae

train_total_data, _, _, _, test_data, test_labels = input_data.prepare_MNIST_data()

train_size = 10000
IMAGE_SIZE_MNIST = 28
num_hidden = 500
dim_img = IMAGE_SIZE_MNIST ** 2
dim_z = 2
learn_rate = 1e-3
batch_size = min(128, train_size)
num_epochs = 10
embedding_size = 16*16
LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")

y_input = tf.placeholder(dtype, shape=[None, dim_img], name='input_img')
y_output_true = tf.placeholder(dtype, shape=[None, dim_img], name='target_img')

# network architecture
ae = vae(
    y_input=y_input,
    dim_img=dim_img,
    dim_z=dim_z,
    num_hidden=num_hidden
)

# optimization
with tf.name_scope("Training"):
    train_step = tf.train.AdamOptimizer(learn_rate).minimize((ae.loss))

y_train = train_total_data[:train_size, :-input_data.NUM_LABELS]
print("Num data points", train_size)
print("Num epochs", num_epochs)

merged_summ = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=2)
writer = tf.summary.FileWriter("/tmp/initial/")

with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    session.graph.finalize()
    itr=1
    for epoch in range(num_epochs):
        for i, batch_indices in enumerate(batch_index_groups(batch_size=batch_size, num_samples=train_size)):
            batch_xs_input = y_train[batch_indices, :]
            _, tot_loss, loss_likelihood, loss_divergence = session.run(
                (
                    train_step,
                    ae.loss,
                    ae.neg_marginal_likelihood,
                    ae.kl_divergence
                ),
                feed_dict={
                    y_input: batch_xs_input
                }
            )
            print(
                "epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                 epoch,
                 tot_loss,
                 loss_likelihood,
                 loss_divergence
                )
            )

