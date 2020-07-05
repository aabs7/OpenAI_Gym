from agent import BaseAgent
from mountain_car_tc import MountainCarTileCoder
import numpy as np
from utils import argmax

class SarsaAgent(BaseAgent):
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.gamma = None
        self.iht_size = None
        self.w = None
        self.alpha = None
        self.num_tilings = None
        self.num_tiles = None
        self.mctc = None
        self.initial_weights = None
        self.num_actions = None
        self.previous_tiles = None
        self.sum_reward = 0

    def agent_init(self,agent_info={}):
        self.num_tilings = agent_info.get("num_tilings",32)
        self.num_tiles = agent_info.get("num_tiles",4)
        self.iht_size = agent_info.get("iht_size",4096)
        self.epsilon = agent_info.get("epsilon",0.0)
        self.gamma = agent_info.get("gamma",1.0)
        self.alpha = agent_info.get("alpha",0.5) / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        self.num_actions = agent_info.get("num_actions",3)

        # We need to initialize self.w to three times the iht_size. 
        # This is because we need to have one set of weights for each action
        
        self.w = np.ones((self.num_actions,self.iht_size)) * self.initial_weights

        # We initialize self.mctc to the mountain car version of the tile coder
        self.tc = MountainCarTileCoder(iht_size = self.iht_size,
                                        num_tilings = self.num_tilings,
                                        num_tiles = self.num_tiles)
        
    def select_action(self,tiles):
        action_values = []
        chosen_action = None

        action_values = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            action_values[action] = self.w[action][tiles].sum()

        if(np.random.random() >= self.epsilon):
            chosen_action = argmax(action_values)
        else:
            chosen_action = np.random.randint(self.num_actions)

        return chosen_action,action_values[chosen_action]

    def agent_start(self,state):
        self.sum_reward = 0

        position,velocity = state

        active_tiles = self.tc.get_tiles(position,velocity)
        current_action,action_value = self.select_action(active_tiles)

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_step(self,reward,state):
        
        position,velocity = state

        active_tiles = self.tc.get_tiles(position,velocity)
        current_action,action_value = self.select_action(active_tiles)

        prev_action_value = self.w[self.last_action][self.previous_tiles].sum()
        prev_grad = np.zeros_like(self.w)
        prev_grad[self.last_action][self.previous_tiles] = 1
        #Use SARSA 
        delta = reward + self.gamma * action_value - prev_action_value
        self.w += self.alpha * delta * prev_grad

        self.sum_reward += reward
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_end(self,reward):

        prev_action_value = self.w[self.last_action][self.previous_tiles].sum()
        prev_grad = np.zeros_like(self.w)
        prev_grad[self.last_action][self.previous_tiles] = 1
        #Use SARSA 
        delta = reward - prev_action_value
        self.w += self.alpha * delta * prev_grad

        self.sum_reward += reward

    def agent_cleanup(self):
        pass
    
    def agent_message(self, message):
        if message == "get_reward":
            return self.sum_reward
        else:
            raise Exception("Unrecognized Message!")