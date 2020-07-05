from rl_glue import RLGlue
from mountain_car_env import MountainCarEnvironment
from sarsa_agent import SarsaAgent

num_runs = 10
num_episodes = 300
env_info = {"num_tiles":8,"num_tilings":8}
agent_info = {}
all_steps = []

agent = SarsaAgent
env = MountainCarEnvironment

for run in range(num_runs):
    rl_glue = RLGlue(env,agent)
    rl_glue.rl_init(agent_info,env_info)

    for episode in range(num_episodes + 1):
        rl_glue.rl_episode(15000)
        r = rl_glue.rl_agent_message("get_reward")
        print("episode:",episode,"reward:",r)

