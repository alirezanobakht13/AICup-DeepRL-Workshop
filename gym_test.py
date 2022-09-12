import gym
import time
env = gym.make("LunarLander-v2")
observation = env.reset()
for _ in range(500):
   env.render()
   action = env.action_space.sample()  # User-defined policy function
   observation, reward, done, info = env.step(action)
   time.sleep(0.02)

   if done:
      observation = env.reset()
env.close()