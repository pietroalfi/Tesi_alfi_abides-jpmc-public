# File to test if the agent integrates well with the simulator
# The agent repeatley takes the action specified in env.step (line 20)
import gym
from gym.envs.registration import register
import abides_gym
from tqdm import tqdm

if __name__ == "__main__":
    env = gym.make(
        "markets-mm-basic-v0",
        #"markets-mm-riskav-v1",
        background_config="rmsc04", #"rmsc04"
    )

    env.seed(0)
    state = env.reset()
    sprint("stato",state)
    for i in tqdm(range(30)):
         
        state, reward, done, info = env.step(5)
        print("stato",state)
        print("reward",reward)
        print("done",done)
        print("info", info)