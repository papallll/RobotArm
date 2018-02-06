import numpy as np
from reinforce import DQN
from env import Arm

def train(env):

	s = env.reset()

	while True:

		env.render()

		action = DQN.choose_action(s)

		state_, reward, done = env.step(action)

		DQN.store(state, action, state_, reward)

		DQN.learn()

		s = s_

		if done:
			break

	
		

if __name__ == '__main__':

	env = Arm()
	train(env)