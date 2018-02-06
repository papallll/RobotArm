import pyglet, time
import numpy as np

class Arm(object):
	"""docstring for Arm"""
	def __init__(self):
		super(Arm, self).__init__()
		self.viewer = None
		self.action = np.zeros([2])
		self.state = np.zeros([2])
		self.target_in = False
		self.counter = 0
		self.target_cor = [100,300]
		# self.target_cor = 400 * np.random.rand(2)
		pyglet.clock.set_fps_limit(30)

		self.arm_length = 100
		self.arm_width = 5
		self.height = 400
		self.width = 400
		self.center = [int(self.height / 2), int(self.width / 2)]

	def step(self, action):

		self.action = action * 0.05
		self.state += self.action
		self.state = self.state % 2
		# print(self.state)
		
		reward, done, tmp = self.reward()

		return np.hstack([self.state, tmp, done*1]), reward, done

	def reset(self):
		
		self.action = np.zeros([2])
		self.state = np.zeros([2])
		self.target_in = False
		self.counter = 0
		# self.target_cor = [300,300]
		self.target_cor = 200 * np.random.rand(2) + 100
		
		return np.zeros([7])

	def render(self):
		
		if self.viewer is None:
			self.viewer = Viewer(self.arm_length, self.arm_width, self.center, self.width, self.height, self.target_cor)
		self.viewer.render(self.state, self.target_cor)

	def reward(self):

		angle1 = self.state[0] * np.pi
		angle2 = self.state[1] * np.pi
		top_cor = np.array([self.center[0] + self.arm_length * np.sin(angle1) + self.arm_length * np.sin(angle2), \
							self.center[1] + self.arm_length * np.cos(angle1) + self.arm_length * np.cos(angle2)])
		middle_cor = np.array([self.center[0] + self.arm_length * np.sin(angle1), \
							   self.center[1] + self.arm_length * np.cos(angle1)])

		distance = np.sqrt(np.sum(np.square(np.array(self.viewer.target_cor) - top_cor)))
		if distance > 15:
			done = False
			self.r = - distance / 200
			self.counter = 0
		else:
			if not self.counter:
				self.r = 1 - distance / 200
				self.counter += 1
				done = False
			else:
				self.r += 1
				self.counter += 1
				done = False
				if self.counter >= 50:
					self.r += 100
					done = True
		return self.r, done, np.hstack([(top_cor - self.target_cor) / 200, (np.array(self.target_cor) - self.center) / 200])


class Viewer(pyglet.window.Window):
	"""docstring for Viewer"""
	def __init__(self, arm_length, arm_width, center, width, height, target_cor):

		super(Viewer, self).__init__(width=width, height=height, vsync=False)
		self.center1 = center
		self.center2 = [center[0], center[1] + arm_length]
		self.arm_length = arm_length
		self.arm_width = arm_width
		self.angle = [0, 0]
		self.mouse_in = False

		self.target_cor = target_cor
		# print(target_cor)

		self.batch = pyglet.graphics.Batch()
		pyglet.gl.glClearColor(1,1,1,1)

	def update_arm(self):

		self.batch = pyglet.graphics.Batch()

		self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS,
    None,
    ('v2f', (self.center1[0] - self.arm_width * np.cos(self.angle[0]) , self.center1[1] + self.arm_width * np.sin(self.angle[0]),
             self.center1[0] + self.arm_length * np.sin(self.angle[0]) - self.arm_width * np.cos(self.angle[0]) , self.center1[1] + self.arm_length * np.cos(self.angle[0]) + self.arm_width * np.sin(self.angle[0]),
             self.center1[0] + self.arm_length * np.sin(self.angle[0]) + self.arm_width * np.cos(self.angle[0]) , self.center1[1] + self.arm_length * np.cos(self.angle[0]) - self.arm_width * np.sin(self.angle[0]),
             self.center1[0] + self.arm_width * np.cos(self.angle[0]), self.center1[1] - self.arm_width * np.sin(self.angle[0]))),
     ('c3B', (86, 109, 249) * 4))

		self.center2 = [self.center1[0] + self.arm_length * np.sin(self.angle[0]), self.center1[1] + self.arm_length * np.cos(self.angle[0])]
		self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS,
    None,
    ('v2f', (self.center2[0] - self.arm_width * np.cos(self.angle[1]) , self.center2[1] + self.arm_width * np.sin(self.angle[1]),
             self.center2[0] + self.arm_length * np.sin(self.angle[1]) - self.arm_width * np.cos(self.angle[1]) , self.center2[1] + self.arm_length * np.cos(self.angle[1]) + self.arm_width * np.sin(self.angle[1]),
             self.center2[0] + self.arm_length * np.sin(self.angle[1]) + self.arm_width * np.cos(self.angle[1]) , self.center2[1] + self.arm_length * np.cos(self.angle[1]) - self.arm_width * np.sin(self.angle[1]),
             self.center2[0] + self.arm_width * np.cos(self.angle[1]), self.center2[1] - self.arm_width * np.sin(self.angle[1]))),
     ('c3B', (86, 109, 249) * 4))

	def update_target(self):

		x = self.target_cor[0]
		y = self.target_cor[1]

		self.target = self.batch.add(4, pyglet.gl.GL_QUADS,
    None,
    ('v2f', (x - 10, y - 10,
             x + 10, y - 10,
             x + 10, y + 10,
             x - 10, y + 10)),
     ('c3B', (249, 86, 86) * 4))


	def render(self, state, target_cor):

		self.angle = state * np.pi
		
		self.update_arm()
		if self.mouse_in:
			self.update_target()	
		else:
			self.target_cor = target_cor
			self.update_target()
		self.switch_to()
		self.dispatch_events()
		self.dispatch_event('on_draw')
		self.flip()

	def on_draw(self):

		self.clear()
		self.batch.draw()

	def on_mouse_motion(self, x, y, dx, dy):

		self.target_cor = [x, y]

	def on_mouse_enter(self, x, y):
		self.mouse_in = True

	def on_mouse_leave(self, x, y):
		self.mouse_in = False

if __name__ == '__main__':

	env = Arm()
	env.render()
	while True:
		env.step(2*np.random.rand(2)-1)
		env.render()
		# time.sleep(1)