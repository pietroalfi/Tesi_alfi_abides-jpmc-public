# File for Market Maker agent's training (via DQN)
import ray
from ray import tune

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
from abides_gym.envs.markets_mm_custom_metrics import MyBaseCallbacks

api_key = wandb.api.api_key

# Example with custom callbacks and WandB

# Import to register environments
import abides_gym

from ray.tune.registry import register_env

# import env

from abides_gym.envs.markets_mm_basic_environment_v0 import (
    SubGymMarketsMmBasicEnv_v0,
)

register_env(
    "markets-mm-basic-v0",
    lambda config: SubGymMarketsMmBasicEnv_v0(**config),
)

ray.shutdown()
ray.init(num_cpus=6)

"""
PPO's default:
sample_batch_size=200, batch_mode=truncate_episodes, training_batch_size=4000 -> workers collect batches of 200 ts (per worker), then the policy network gets updated using the 4000 last ts (in n minibatch-chunks of 128 (sgd_minibatch_size)).

DQN's default:
train_batch_size=32, sample_batch_size=4, timesteps_per_iteration=1000 -> workers collect chunks of 4 ts and add these to the replay buffer (of size buffer_size ts), then at each train call, at least 1000 ts are pulled altogether from the buffer (in batches of 32) to update the network.
"""

name_xp = "dqn_mm_basic_v0"
tune.run(
    "DQN",
    name=name_xp,
    resume=False,
    stop={"training_iteration": 100}, 
    checkpoint_at_end=True, 
    checkpoint_freq=20, 
    config={
        "callbacks": MyBaseCallbacks,
        "env": "markets-mm-basic-v0",
        "env_config": {
            "mkt_close": "16:00:00",
            "timestep_duration": "60S", #tune.grid_search(["30S", "60S"]),
            "order_fixed_size": 100, #tune.grid_search([200, 500, 1000]),
        },
        "seed": 1,#tune.grid_search([1, 2, 3]),
        "num_gpus": 0,
        "num_workers": 0,
        "hiddens": [128, 64, 32], #[50, 20]
        "gamma": 0.999, #1
        "lr": 0.001, # 0.0001
        "framework": "torch",
        "observation_filter": "MeanStdFilter",
    },
    callbacks=[
        WandbLoggerCallback(
            project="abides_market_maker_00_abm",
            group=name_xp,
            api_key=api_key,
            log_config=False,
        )
    ],
)