import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np


# ---------------------------------------------------------------------------- #
#                                Neural Network                                #
# ---------------------------------------------------------------------------- #
class DQN(nn.Module):
    def __init__(self,input_dim,l1_dim,l2_dim,output_dim,lr):
        super().__init__()

        self.input_dim = input_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.output_dim = output_dim

        self.l1 = nn.Linear(input_dim,l1_dim)
        self.l2 = nn.Linear(l1_dim,l2_dim)
        self.l3 = nn.Linear(l2_dim,output_dim)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),lr=lr)

    def forward(self, state):
        layer1 = self.l1(T.tensor(state,dtype=T.float32))
        layer1_activated = F.relu(layer1)
        layer2 = self.l2(layer1_activated)
        layer2_activated = F.relu(layer2)
        output = self.l3(layer2_activated)

        return output


# ---------------------------------------------------------------------------- #
#                                     Agent                                    #
# ---------------------------------------------------------------------------- #
class Agent:
    def __init__(self,input_dim,l1_dim,l2_dim,output_dim,lr,gamma,epsilon,eps_dec_rate,eps_min,
        batch_size,max_replay_memory):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec_rate = eps_dec_rate
        self.eps_min = eps_min

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.batch_size = batch_size
        self.max_replay_memory = max_replay_memory

        self.brain = DQN(input_dim,l1_dim,l2_dim,output_dim,lr)

        self.memory_init()
    
    def memory_init(self):
        self.states = np.zeros((self.max_replay_memory,self.input_dim))
        self.actions = np.zeros(self.max_replay_memory,dtype=np.int32)
        self.rewards = np.zeros(self.max_replay_memory)
        self.next_states = np.zeros((self.max_replay_memory,self.input_dim))
        self.dones = np.zeros(self.max_replay_memory,dtype=np.bool)

        self.counter = 0

    def store(self,state,action,reward,next_state,done):
        temp = self.counter % self.max_replay_memory
        self.counter += 1

        self.states[temp] = state
        self.actions[temp] = action
        self.rewards[temp] = reward
        self.next_states[temp] = next_state
        self.dones[temp] = done
    
    def sample(self):
        size = min(self.counter,self.max_replay_memory)
        indices = np.random.choice(size,self.batch_size)

        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.dones[indices]

        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

    def choose_action(self,state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.output_dim)
        else:
            with T.no_grad():
                out = self.brain(state)
            return out.argmax().item()
    
    def epsilon_decay(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec_rate

    
    def learn(self):
        if self.counter < self.batch_size:
            return 0
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample()

        
        q_now = self.brain(state_batch)

        with T.no_grad():
            q_target = self.brain(state_batch)
            q_next = self.brain(next_state_batch)
        

        q_next[done_batch] = 0.0

        indices = np.arange(self.batch_size,dtype=np.int32)

        # print(q_target[indices,action_batch].shape)

        q_target[indices,action_batch] = T.tensor(reward_batch,dtype=T.float32) + self.gamma * q_next.max(dim=1)[0]

        self.brain.optimizer.zero_grad()
        loss = self.brain.loss(q_target,q_now)
        loss.backward()
        self.brain.optimizer.step()

        return float(T.mean(loss))     



# ---------------------------------------------------------------------------- #
#                                     Train                                    #
# ---------------------------------------------------------------------------- #
import gym
import time
env = gym.make("CartPole-v1")

# input_dim,l1_dim,l2_dim,output_dim,lr,gamma,epsilon,eps_dec_rate,eps_min,
#         batch_size,max_replay_memory

agent = Agent(*env.observation_space.shape,64,128,env.action_space.n,0.0001,0.99,1,0.0009,0.01,
        128,100000)

EPISODS = 500

rewards = []
costs = []
epsilons = []


for i in range(EPISODS):
    state = env.reset()
    step_counter = 0
    total_cost = 0
    total_reward = 0

    while True:
        step_counter +=1
        env.render()

        action = agent.choose_action(state)
        # if i > 500:
        #     print(action)

        next_state,reward,done,_ = env.step(action)

        agent.store(state,action,reward,next_state,done)
        total_reward += reward

        total_cost += agent.learn()

        state = next_state

        if done:
            break

    
    epsilons.append(agent.epsilon)
    costs.append(total_cost/step_counter)
    rewards.append(total_reward)

    if i%10 == 0:
        print(f"episod {i}, reward {total_reward}, epsilon {epsilons[-1]}")

    agent.epsilon_decay()


env.close()


# ---------------------------------------------------------------------------- #
#                                     Plot                                     #
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,5))

# rewards = np.array([1,2,3,4,5,6,7,8,9,10])
# epsilons = rewards
# costs = rewards[::-1]

x_reward = [i for i in range(len(rewards))]
x_epsilon = [i for i in range(len(epsilons))]
x_cost = [i for i in range(len(costs))]

plt1 = fig.add_subplot(1,3,1)
plt1.scatter(x_reward,rewards,color = "C1")
plt1.set_xlabel("Episod",color="C1")
plt1.set_ylabel("Rewards", color = "C1")

plt2 = fig.add_subplot(1,3,2)
plt2.plot(x_epsilon,epsilons,color = "C2")
plt2.set_xlabel("Episod",color = "C2")
plt2.set_ylabel("Epsilon", color = "C2")

plt3 = fig.add_subplot(1,3,3)
plt3.plot(x_cost,costs,color = "C3")
plt3.set_xlabel("Episod",color = "C3")
plt3.set_ylabel("Cost", color = "C3")


fig.savefig("sssplots.jpg")


T.save(agent.brain,"trained_agent_brain_LunarLander_no_zip.pt")


# ---------------------------------------------------------------------------- #
#                              Load trained agent                              #
# ---------------------------------------------------------------------------- #
env = gym.make("CartPole-v1")

## Brain related parameters are not important here.
trained_agent = Agent(*env.observation_space.shape,2048,1024,env.action_space.n,0.0001,0.99,1,0.0005,0.01,
        256,100000)

## load brain:
trained_agent.brain = T.load("trained_agent_brain_LunarLander_no_zip.pt").cpu()

## fully greedy selection or epsilon greedy:
FULLY_GREEDY = True

if FULLY_GREEDY:
    trained_agent.epsilon = 0.0
else:
    trained_agent.epsilon = 0.01


# ---------------------------------------------------------------------------- #
#                                     Test                                     #
# ---------------------------------------------------------------------------- #
EPISODS = 5

for i in range(EPISODS):
    state = env.reset()
    step_counter = 0
    total_cost = 0
    total_reward = 0

    while True:
        step_counter +=1

        env.render()

        time.sleep(0.025)

        action = trained_agent.choose_action(state)
        # if i > 500:
        #     print(action)

        next_state,reward,done,_ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    print(f"episod {i}, reward {total_reward}")
env.close()