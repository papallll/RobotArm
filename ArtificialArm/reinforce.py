import tensorflow as tf
import numpy as np
from env import Arm


np.random.seed(1)
tf.set_random_seed(1)

class Actor(object):
	"""docstring for Actor"""
	def __init__(self, sess, mode):
		super(Actor, self).__init__()

		self.learn_counter = 0
		self.iter_max = 1100
		self.lr = 1e-4
		self.sess = sess
		self.epsilon = 0.1
		self.mode = mode
		self.decay = 0.9999
		
		self.action_shape = 2
		self.state_shape = 7

		self.s = tf.placeholder(tf.float32, [None, self.state_shape], 's')
		self.s_ = tf.placeholder(tf.float32, [None, self.state_shape], 's_')

		with tf.variable_scope('actor'):
			self.a = self._built_net(self.s, 'eval', trainable=True)
			self.a_ = self._built_net(self.s_, 'tar', trainable=False)

		self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
		self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/tar')

		self.resign_op = [tf.assign(x, y) for x, y in zip(self.t_params, self.e_params)]

	def _built_net(self, input_, scope, trainable):

		with tf.variable_scope(scope):
			w_init = tf.contrib.layers.xavier_initializer()
			init_b = tf.constant_initializer(0.01)

			net_shape = [200, 200, 10]

			with tf.variable_scope('net'):
				l1 = tf.layers.dense(inputs=input_, units=net_shape[0], activation=tf.nn.relu6, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
				l2 = tf.layers.dense(inputs=l1, units=net_shape[1], activation=tf.nn.relu6, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
				l3 = tf.layers.dense(inputs=l2, units=net_shape[2], activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
				output = tf.layers.dense(inputs=l3, units=self.action_shape, activation=tf.nn.tanh, kernel_initializer=w_init, trainable=trainable)

			return output

	def choose_action(self, s):

		# if self.mode == 'train':
		# 	if np.random.rand() > self.epsilon:
		s = s[np.newaxis, :]
		self.out = np.clip(np.random.normal(self.sess.run(self.a, feed_dict={self.s:s}), max(2, self.decay*self.epsilon)), *[-1,1])
		# 	else:
		# 		self.out = 2*np.random.rand(2) - 1

		# else:
		# 		s = s[np.newaxis, :]
		# 		self.out = self.sess.run(self.a, feed_dict={self.s:s})
		# print(self.out)

		return self.out

	def learn(self, s):
		self.sess.run(self.train_op, feed_dict={self.s:s})
		if self.learn_counter % self.iter_max == 0:
			self.sess.run(self.resign_op)
		self.learn_counter += 1
		self.epsilon = self.decay*self.epsilon

	def cal_grad(self, grad_c):
		opt = tf.gradients(xs=self.e_params, ys=self.a, grad_ys=grad_c)
		self.train_op = tf.train.RMSPropOptimizer(-self.lr).apply_gradients(zip(opt, self.e_params))
		

class Critic(object):
	"""docstring for Critic"""
	def __init__(self, sess, s, s_, a, a_):
		super(Critic, self).__init__()

		self.learn_counter = 0
		self.iter_max = 1000
		self.lr = 1e-4
		self.gama = 0.9
		self.sess = sess

		self.action_shape = 2
		self.state_shape = 7

		self.s = s
		self.s_ = s_
		self.a = a
		self.a_ = a_

		# self.sa = tf.concat([self.s, self.a], axis=1, name='sa')
		# self.sa_ = tf.concat([self.s_, self.a_], axis=1, name='sa_')
		self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

		with tf.variable_scope('critic'):
			self.q = self._built_net(self.s, self.a, 'eval', trainable=True)
			self.q_ = self._built_net(self.s_, self.a_, 'tar', trainable=False)

		self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
		self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/tar')

		self.resign_op = [tf.assign(x, y) for x, y in zip(self.t_params, self.e_params)]

		self.grad_c = tf.gradients(xs=self.a, ys=self.q)

		self.traget_q = self.reward + self.gama * self.q_
		loss = tf.reduce_mean(tf.square(self.traget_q - self.q))
		self.train_opt = tf.train.RMSPropOptimizer(self.lr).minimize(loss)

	def _built_net(self, input_s, input_a, scope, trainable):

		with tf.variable_scope(scope):
			w_init = tf.contrib.layers.xavier_initializer()
			init_b = tf.constant_initializer(0.01)

			net_shape = [200, 200, 10]

			ls = tf.layers.dense(inputs=input_s, units=net_shape[0], activation=tf.nn.relu6, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
			la = tf.layers.dense(inputs=input_a, units=net_shape[0], activation=tf.nn.relu6, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
			
			l2 = tf.layers.dense(inputs=(ls + la), units=net_shape[1], activation=tf.nn.relu6, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
			l3 = tf.layers.dense(inputs=l2, units=net_shape[2], activation=tf.nn.relu6, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)
			output = tf.layers.dense(inputs=l3, units=1, kernel_initializer=w_init, bias_initializer=init_b, trainable=trainable)

		return output

	def learn(self, s, s_, r):

		_, t = sess.run([self.train_opt, self.traget_q], feed_dict={self.s:s, self.s_:s_, self.reward:r})
		# print(t)
		if self.learn_counter % self.iter_max == 0:
			sess.run(self.resign_op)
			# print('===resigned===')
		self.learn_counter += 1

class Memory(object):
	"""docstring for Memory"""
	def __init__(self):
		super(Memory, self).__init__()
		self.s = np.zeros([7])
		self.s_ = np.zeros([7])
		self.r = np.zeros([1])
		self.memory_max = 5000
		self.isfull = False

	def add(self, s, s_, r):

		self.s = np.vstack((self.s, s))
		# print((self.s_, s_))
		self.s_ = np.vstack((self.s_, s_))
		self.r = np.vstack((self.r, r))

		if self.s.shape[0] == self.memory_max:
			self.isfull = True

		self.s = self.s[:self.memory_max, :]
		self.s_ = self.s_[:self.memory_max, :]
		self.r = self.r[:self.memory_max, :]

	def Sample(self, batch_size):

		index = np.random.choice(self.memory_max, batch_size)
		return self.s[index,:], self.s_[index,:], self.r[index,:]

	def reset(self):
		self.s = np.array([])
		self.s_ = np.array([])
		self.isfull = False		

		
if __name__ == '__main__':

	env = Arm()
	train_time = 500
	# iter_max = 3000
	batch_size = 32
	step_max = 200
	mode = 'tran'

	sess = tf.Session()

	if mode == 'train':
		act = Actor(sess, mode=mode)
		cri = Critic(sess, act.s, act.s_, act.a, act.a_)
		act.cal_grad(cri.grad_c)

		sess.run(tf.global_variables_initializer())

		mem = Memory()
		for i in range(train_time):
			s = env.reset()
			
			step = 0

			while True:
				
				step += 1
				# print(s)
				env.render()
				a = act.choose_action(s)
				s_, r, done = env.step(a.ravel())
				print(a)
				# time.sleep(1)
				mem.add(s, s_, r)

				if mem.isfull:
					# print('========')
					sample_s, sample_s_, sample_r = mem.Sample(batch_size)
					cri.learn(sample_s, sample_s_, sample_r)
					act.learn(sample_s)

				if done or step > step_max:
					break

				s = s_
				# print(s)

		saver = tf.train.Saver()
		saver_path = saver.save(sess, "save/model.ckpt")
		print("Model saved in file:", saver_path)

	else:
		
		act = Actor(sess, mode=mode)
		cri = Critic(sess, act.s, act.s_, act.a, act.a_)
		act.cal_grad(cri.grad_c)
		saver = tf.train.Saver()
		saver.restore(sess, "save/model.ckpt")
		s = env.reset()
		while True:

			env.render()
			a = act.choose_action(s)
			s_, r, done = env.step(a.ravel())
			# print(r)
			s = s_