import ray
from ray import tune

from ray.tune.logger import DEFAULT_LOGGERS
#from ray.tune.integration.wandb import WandbLoggerCallback

#import wandb
#from abides_gym.envs.markets_execution_custom_metrics import MyCallbacks

# Example to run RLlib in simple mode
# no custom metrics/callbacks or WandB

# Import to register environments
import abides_gym

from ray.tune.registry import register_env

# import env

from abides_gym.envs.markets_execution_environment_v0 import (
    SubGymMarketsExecutionEnv_v0,
)

register_env(
    "markets-execution-v0",
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)

ray.shutdown()
ray.init()

"""
PPO's default:
sample_batch_size=200, batch_mode=truncate_episodes, training_batch_size=4000 -> workers collect batches of 200 ts (per worker), then the policy network gets updated using the 4000 last ts (in n minibatch-chunks of 128 (sgd_minibatch_size)).

DQN's default:
train_batch_size=32, sample_batch_size=4, timesteps_per_iteration=1000 -> workers collect chunks of 4 ts and add these to the replay buffer (of size buffer_size ts), then at each train call, at least 1000 ts are pulled altogether from the buffer (in batches of 32) to update the network.
"""

name_xp = "dqn_execution_v1_10"
tune.run(
    "DQN",
    name=name_xp,
    resume=False,
    stop={"training_iteration": 20},  # "episode_reward_mean": 2e6,
    checkpoint_at_end=True, 
    checkpoint_freq=20,
    config={
        "env": "markets-execution-v0",
        "env_config": {
            "mkt_close": "16:00:00",
            "timestep_duration": "30S", #tune.grid_search(["30S", "60S"]),
            "execution_window": "01:00:00", #tune.grid_search(["01:00:00", "00:30:00"]),
            "parent_order_size": 10000,#tune.grid_search([10000, 20000, 30000]),
            "order_fixed_size": 200,#tune.grid_search([200, 500, 1000]),
            "not_enough_reward_update": 0,# tune.grid_search([0, -100]),
#            "oracle_parameters": {
#                "l_1": 100,#tune.grid_search([-100, 0, 100]),
#               "sin_amp": 100,#tune.grid_search(
#                    #[
#                    #   0,
#                    #    100,
#                    #]
#                #),
#                "sin_freq":2,# tune.grid_search([2, 10, 50]),
#                "l_2": 0,#tune.grid_search([0, -100]),
#                "sigma":50, # tune.grid_search([0, 50]),
#            },
        },
        "seed": 1,#tune.grid_search([1, 2, 3]),
        "num_gpus": 0,
        "num_workers": 0,
        "hiddens": [50, 20],
        "gamma": 1,
        "lr": 0.0001,
        "framework": "torch",
        "observation_filter": "MeanStdFilter",
    },
)