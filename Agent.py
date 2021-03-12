import numpy as np
import time
NUM_DIZITIZED = 6
ETA = 0.5
GAMMA = 0.99
MAX_STEPS = 1000
NUM_EPISODES = 1000
D2R = 3.141592/180
class Agent:
	
	def __init__(self,num_states,num_actions):
		self.brain = Brain(num_states,num_actions)
	def update_Q_function(self,observation,action,reward,observation_next):
		self.brain.update_Q_table(observation,action,reward,observation_next)
	def get_action(self,observation,step):
		action = self.brain.decide_action(observation,step)
		return action

class Brain:
	def __init__(self,num_states,num_actions):
		self.num_actions = num_actions
		self.q_table = np.random.uniform(low=0,high=1,size=(NUM_DIZITIZED**num_states,num_actions))
	def bins(self,clip_min,clip_max,num):
		return np.linspace(clip_min,clip_max,num+1)[1:-1]
	def digitize_state(self,observation):
		cart_pos,cart_v,pole_angle,pole_v = observation
		digitized = [np.digitize(cart_pos,bins =self.bins(-2.4,2.4,NUM_DIZITIZED)),
		np.digitize(cart_v,bins =self.bins(-3.0,3.0,NUM_DIZITIZED)),
		np.digitize(pole_angle,bins =self.bins(-0.5,0.5,NUM_DIZITIZED)),
		np.digitize(pole_v,bins =self.bins(-2.0,2.0,NUM_DIZITIZED))]
		return sum([x*(NUM_DIZITIZED**i) for i, x in enumerate(digitized)])
	def update_Q_table(self,observation,action,reward,observation_next):
		state = self.digitize_state(observation)
		state_next = self.digitize_state(observation_next)
		Max_Q_next = max(self.q_table[state_next][:])
		self.q_table[state,action] = self.q_table[state,action]+ETA*(reward+GAMMA*Max_Q_next - self.q_table[state,action])
	def decide_action(self,observation,episode):
		state = self.digitize_state(observation)
		epsilon = 0.5*(1/(episode+1))
		if epsilon <= np.random.uniform(0,1):
			action = np.argmax(self.q_table[state][:])
		else:
			action = np.random.choice(self.num_actions)
		return action

class Environment:
	def __init__(self):
		import pybullet as p
		import time
		import pybullet_data
		import numpy as np
		import math
		useMaximalCoordinates = False
		self.p = p
		self.p.connect(self.p.GUI)
		self.p.setGravity(0, 0, 0)
		self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.pole = self.p.loadURDF("cartpole.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)
		for i in range(self.p.getNumJoints(self.pole)):
			self.p.setJointMotorControl2(self.pole, i, p.POSITION_CONTROL, targetPosition=0, force=0)

		useRealTimeSim = False
		self.p.setRealTimeSimulation(useRealTimeSim)
		numJoints = self.p.getNumJoints(self.pole)
		self.q = [0,0]
		self.qdot = [0,0]
		self.timeStep = 0.001
		self.count = 0;
		self.maxF = [20,20];
		num_states = 4
		num_actions = 2
		
		self.agent = Agent(num_states,num_actions)
	def step(self,action):
		done = False
		if action == 1:
			self.p.setJointMotorControlArray(self.pole, [0,1], controlMode=self.p.TORQUE_CONTROL, forces=[self.maxF[0],0])
		else:
			self.p.setJointMotorControlArray(self.pole, [0,1], controlMode=self.p.TORQUE_CONTROL, forces=[-self.maxF[0],0])
		self.p.stepSimulation()
		time.sleep(self.timeStep)
		jointStates = self.p.getJointStates(self.pole,[0,1])
		self.q = np.array([jointStates[0][0],jointStates[1][0]])
		self.qdot = np.array([jointStates[0][1],jointStates[1][1]])
		observation  = [self.q[0],self.qdot[0],self.q[1],self.qdot[1]]
		#print(observation)
		if abs(self.q[0])<=2.4:
			if abs(self.q[1])<=20.9*D2R:
				reward = 1;
			elif abs(self.q[1])>20.9*D2R:
				reward = 0;

		elif abs(self.q[0])>2.4:
			reward = 0;


		if self.count > 200:
			done = True;
		if reward == 0:
			done = True;
		self.count = self.count+1;
		info = 0
		return observation,reward,done,info

	def run(self):
		complete_episodes = 0
		is_episode_final = False
		save_q = []
		for episode in range(NUM_EPISODES):	
			observation = [0,0,0,0]
			self.p.resetJointState(self.pole,0,(np.random.rand(1)-0.5)/2)
			self.p.resetJointState(self.pole,1,(np.random.rand(1)-0.5)/2)

			self.count = 0;
			for step in range(MAX_STEPS):
				if is_episode_final is True:
					save_q.append(observation)
				action = self.agent.get_action(observation,episode)
				observation_next,_,done,_ = self.step(action)

				if done:
					if step < 195:
						reward = -1
						complete_episodes = 0
					elif step >= 195:
						reward = 1
						complete_episodes += 1
				else:
					reward = 0
				print("Episode : ",episode,"count : ",self.count,"step : ",step,"done : ",done,"reward : ",reward,"complete : ",complete_episodes)
				self.agent.update_Q_function(observation,action,reward,observation_next)
				observation = observation_next
				if done:
					print('##########{0} Episode : Finished after {1} time steps###########'.format(episode,step+1))
					if is_episode_final is False:
						break;
			if is_episode_final is True:
				np.save("save_q.npy",save_q)
				break;
			if complete_episodes >= 20:
				print("10 Episode Complete")
				is_episode_final = True

		





