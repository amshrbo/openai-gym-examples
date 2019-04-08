import gym

env = gym.make('CartPole-v0') # first create our environment
env.reset() # reset our env default values

# rendring our env
for _ in tuple(range(1000)):
    env.render()
    env.step(env.action_space.sample()) # take random actions at first