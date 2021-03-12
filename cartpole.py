import pybullet as p
import time
import pybullet_data
import numpy as np
import math
useMaximalCoordinates = False
p.connect(p.GUI)
p.setGravity(0, 0, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pole = p.loadURDF("cartpole.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)

for i in range(p.getNumJoints(pole)):
  p.setJointMotorControl2(pole, i, p.POSITION_CONTROL, targetPosition=0, force=0)

timeStepId = p.addUserDebugParameter("timeStep", 0.001, 0.1, 0.01)
useRealTimeSim = False
p.setRealTimeSimulation(useRealTimeSim)


numJoints = p.getNumJoints(pole)

desired_q = np.array([2,0]);
desired_qdot = np.array([0,0]);
kpCart = 500
kdCart = 150
kpPole = 10
kdPole = 1
prev_q = 0
D2R = math.pi/180
kps = [kpCart, kpPole]
kds = [kdCart, kdPole]
maxF = [10,1000]
def PDcontrol(q,qdot):

	qError = desired_q - q;
	qdotError = desired_qdot - qdot;
	Kp = np.diagflat(kps)
	Kd = np.diagflat(kds)
	forces = Kp.dot(qError)+Kd.dot(qdotError)
	force1 = np.clip(forces[0],-maxF[0],maxF[0])
	force2 = np.clip(forces[1],-maxF[0],maxF[1])
	forces = [force1,force2]
	p.setJointMotorControlArray(pole, [0,1], controlMode=p.TORQUE_CONTROL, forces=forces)







def digitize_state(observation):
	NUM_DIZITIZED = 6
	def bins(clip_min,clip_max,num):
		return np.linspace(clip_min,clip_max,num+1)[1:-1]
	cart_pos,cart_v,pole_angle,pole_v = observation
	digitized = [np.digitize(cart_pos,bins =bins(-2.4,2.4,NUM_DIZITIZED)),
np.digitize(cart_v,bins =bins(-3.0,3.0,NUM_DIZITIZED)),
np.digitize(pole_angle,bins =bins(-0.5,0.5,NUM_DIZITIZED)),
np.digitize(pole_v,bins =bins(-2.0,2.0,NUM_DIZITIZED))]
	return sum([x*(NUM_DIZITIZED**i) for i, x in enumerate(digitized)])




def step(action,q,qdot,count):
	done = False
	if action == 1:
		p.setJointMotorControlArray(pole, [0,1], controlMode=p.TORQUE_CONTROL, forces=[maxF[0],0])
	else:
		p.setJointMotorControlArray(pole, [0,1], controlMode=p.TORQUE_CONTROL, forces=[-maxF[0],0])
	observation  = [q[0],qdot[0],q[1],qdot[1]]
	if abs(q[0])<=2.4:
		if abs(q[1])<20.9*D2R:
			reward = 1;
		else:
			reward = 0;
	else:
		reward = 0;
	if reward ==0 :
		done = True;
	if count > 200:
		done = True;
	
	return observation,reward,done

toggle = 1;
count = 0;
done = False;
while p.isConnected():
	timeStep = p.readUserDebugParameter(timeStepId)
	p.setTimeStep(timeStep)
	jointStates = p.getJointStates(pole,[0,1])
	q = np.array([jointStates[0][0],jointStates[1][0]])
	qdot = np.array((q-prev_q )/timeStep)
	#PDcontrol(q,qdot)
	if toggle == 1:
		observation,reward,done=step(1,q,qdot,count)
	else:
		observation,reward,done=step(0,q,qdot,count)
	print(observation)
	print(digitize_state(observation))
	count = count+1;



	prev_q = q
	if done:
		break;
	
	if (not useRealTimeSim):
		p.stepSimulation()
		time.sleep(timeStep)
