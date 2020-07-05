from rl_glue import RLGlue
from lunar_lander import LunarLanderEnvironment
from sarsa_agent import Agent



experiment_parameters = {
    "num_runs" : 1,
    "num_episodes" : 300,
    "timeout" : 1000
}

environment_parameters = {}

agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}

current_env = LunarLanderEnvironment
current_agent = Agent

rlglue = RLGlue(current_env,current_agent)

env_info = {}
agent_info = agent_parameters

for run in range(1,experiment_parameters["num_runs"] + 1):
    agent_info["seed"] = run
    agent_info["network_config"]["seed"] = run
    env_info["seed"] = run
    
    rlglue.rl_init(agent_info,env_info)

    for episode in range(1,experiment_parameters["num_episodes"]+1):
        rlglue.rl_episode(experiment_parameters["timeout"])
        episode_reward = rlglue.rl_agent_message("get_sum_reward")
        print("episode:",episode," reward:",episode_reward)


