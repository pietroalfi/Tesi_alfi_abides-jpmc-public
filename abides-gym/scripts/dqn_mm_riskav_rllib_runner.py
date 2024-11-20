# File for Market Maker agent's training (via DQN)
import ray
from ray import tune

# With custom callbacks and WandB
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
from abides_gym.envs.markets_mm_custom_metrics import MyBaseCallbacks

api_key = wandb.api.api_key

# Import to register environments
import abides_gym

# import env
from ray.tune.registry import register_env
from abides_gym.envs.markets_mm_riskav_environment_v0 import (
    SubGymMarketsMmRiskAvEnv_v0,
)

register_env(
    "markets-mm-riskav-v0",
    lambda config: SubGymMarketsMmRiskAvEnv_v0(**config),
)

ray.shutdown()
ray.init(num_cpus=6)

"""
DQN's default:
train_batch_size=32, sample_batch_size=4, timesteps_per_iteration=1000 -> workers collect chunks of 4 ts and add these to the replay buffer (of size buffer_size ts), then at each train call, at least 1000 ts are pulled altogether from the buffer (in batches of 32) to update the network.
"""

name_xp = "dqn_mm_riskav_v0"
tune.run(
    "DQN",
    name=name_xp,
    resume=False,
    stop={"training_iteration": 100}, 
    checkpoint_at_end=True, 
    checkpoint_freq=20, 
    config={
        "callbacks": MyBaseCallbacks,
        "env": "markets-mm-riskav-v0",
        #"timesteps_per_iteration": 23400,
        "env_config": {
            "mkt_close": "16:00:00",
            "timestep_duration": "60s", #tune.grid_search(["30S", "60S"]),
            "order_fixed_size": 100, #tune.grid_search([200, 500, 1000]),
        },
        "seed": 1,#tune.grid_search([1, 2, 3]),
        "num_gpus": 0,
        "num_workers": 0,
        "hiddens": [50, 20], #[128, 64, 32],
        "gamma": 0.999, #1
        "lr": 0.001, # 0.0001
        "framework": "torch",
        "observation_filter": "MeanStdFilter",
        "model": {
            "fcnet_hiddens": [50, 20],  
            "fcnet_activation": "relu",  
        },
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