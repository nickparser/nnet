import numpy as np
import tensorflow as tf

class NNet:
	def __init__(
		self,
		n_input,
		n_hidden,
		n_output,
		learning_rate
	):
		self._n_i = n_input
		self._n_h = n_hidden
		self._n_o = n_output
		self._lr = learning_rate

		self._build_net()

	def _build_net(self):
		# declare the training data placeholders / the target data placeholder
		self._i_l = tf.placeholder(tf.float32, [None, self._n_i])
		self._t_l = tf.placeholder(tf.float32, [None, self._n_o])

		# build a neural network
		self._h_l = tf.layers.dense(self._i_l, self._n_h, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
		self._o_l = tf.layers.dense(self._h_l, self._n_o, tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(0, 0.1))

		# the cross entropy cost function / add an optimiser
		self._cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels = self._t_l, logits = self._o_l)
		)
		self._optimizer = tf.train.AdamOptimizer(learning_rate = self._lr).minimize(self._cross_entropy)

		# an accuracy assessment operation
		correct_prediction = tf.equal(tf.argmax(self._t_l, 1), tf.argmax(self._o_l, 1))
		self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# save / restore tf model
		self._saver = tf.train.Saver()

		self._sess = tf.Session()
		self._sess.run(tf.global_variables_initializer())

	def train(self, input_dataset, target_dataset, status):
		self._sess.run([self._optimizer], feed_dict = {self._i_l: input_dataset, self._t_l: target_dataset})

		if status:
			_loss, _accuracy = self._sess.run(
				[self._cross_entropy, self._accuracy],
				feed_dict = {self._i_l: input_dataset, self._t_l: target_dataset}
			)
			print('loss : {:.3f}, accuracy: {:.3f}'.format(_loss, _accuracy))
			

	def predict(self, dataset):
		return self._sess.run(tf.argmax(self._o_l, 1), feed_dict = {self._i_l: dataset})

	def save(self, path):
		self._saver.save(self._sess, path)

	def restore(self, path):
		self._saver.restore(self._sess, path)

