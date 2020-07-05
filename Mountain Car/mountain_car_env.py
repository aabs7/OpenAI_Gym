from environment import BaseEnvironment
import numpy as np 
import gym 

class MountainCarEnvironment(BaseEnvironment):
    def env_init(self,env_info={}):
        self.env = gym.make("MountainCar-v0")
        self.env.seed(0)

    def env_start(self):
        reward = 0.0
        observation = self.env.reset()
        is_terminal = False

        self.reward_obs_term = (reward,observation,is_terminal)
        return self.reward_obs_term[1]


    def env_step(self,action):
        last_state = self.reward_obs_term[1]
        #call gym to do action
        current_state,reward,is_terminal,_ = self.env.step(action)
        self.env.render()
        self.reward_obs_term = (reward,current_state,is_terminal)

        return self.reward_obs_term
        