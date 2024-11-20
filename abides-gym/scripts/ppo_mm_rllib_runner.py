# File for Market Maker agent's training (via PPO)
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
ray.init(num_cpus=4)

"""
PPO's default:
sample_batch_size=200, batch_mode=truncate_episodes, training_batch_size=4000 -> workers collect batches of 200 ts (per worker), then the policy network gets updated using the 4000 last ts (in n minibatch-chunks of 128 (sgd_minibatch_size)).
"""

name_xp = "ppo_mm_basic_v0"  
tune.run(
    "PPO",
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
            "timestep_duration": "60s",  # tune.grid_search(["30S", "60S"]),
            "order_fixed_size": 100,  # tune.grid_search([200, 500, 1000]),
        },
        "seed": 1,  # tune.grid_search([1, 2, 3]),
        "num_gpus": 0,
        "num_workers": 0,
        "gamma": 0.999,  # Discount factor
        "lr": 0.0003,  # Learning rate
        "framework": "torch",
        "observation_filter": "MeanStdFilter",
        "model": {
            "fcnet_hiddens": [50, 20],  # Hidden layers for the policy network
            "fcnet_activation": "relu",  # Activation function
        },
        "kl_coeff": 0.2,
        "clip_param": 0.3,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.0,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
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
