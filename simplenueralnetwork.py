import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def initialize_weights(shape):
	weights =  tf.random_normal(shape, stddev=0.1)
	return tf.Variable(weights)

def fwdpropagation(X,w1,w2):
	h = tf.nn.sigmoid(tf.matmul(X,w1))
	y_= tf.matmul(h,w2)
	return y_

def get_iris_data():
	iris = datasets.load_digits()
	data = iris["data"]
	target = iris["target"]

	N,M = data.shape
	all_X = np.ones((N,M+1))
	all_X[:,1:] = data

	num_labels = len(np.unique(target))
	all_Y = np.eye(num_labels)[target]
	return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
	train_X, test_X, train_y, test_y = get_iris_data()

	x_size=train_X.shape[1]
	h_size= 256 
	y_size=train_y.shape[1]

	X = tf.placeholder("float", shape=[None,x_size])
	y = tf.placeholder("float",shape=[None,y_size])

	w1 = initialize_weights((x_size, h_size))
	w2 = initialize_weights((h_size, y_size))

	y_= fwdpropagation(X,w1,w2)
	predict=tf.argmax(y_,axis=1)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
	updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

	sess = tf.Session()
	init = tf.global_variables_initializer() 
	sess.run(init)

	for j in range(100):
		
		for i in range(len(train_X)):
			sess.run(updates, feed_dict={X: train_X[i: i+1], y:train_y[i: i+1]})

		train_accuracy = np.mean(np.argmax(train_y, axis=1)==sess.run(predict,feed_dict={X: train_X,y: train_y}))
		test_accuracy = np.mean(np.argmax(test_y, axis=1)== sess.run(predict, feed_dict={X: test_X, y: test_y}))

		print 'train',train_accuracy
		print 'test',test_accuracy

	sess.close()
if __name__ == '__main__':
    main()  