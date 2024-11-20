from gym.envs.registration import register
from ray.tune.registry import register_env

from .envs import *


# REGISTER ENVS FOR GYM USE

register(
    id="markets-daily_investor-v0",
    entry_point=SubGymMarketsDailyInvestorEnv_v0,
)

register(
    id="markets-execution-v0",
    entry_point=SubGymMarketsExecutionEnv_v0,
)


register(
    id="markets-mm-basic-v0",
    entry_point='abides_gym.envs:SubGymMarketsMmBasicEnv_v0',
)

register(
    id="markets-mm-basic-v01",
    entry_point='abides_gym.envs:SubGymMarketsMmBasicEnv_v01',
)

register(
    id="markets-mm-riskav-v0",
    entry_point='abides_gym.envs:SubGymMarketsMmRiskAvEnv_v0',
)

register(
    id="markets-mm-riskav-v1",
    entry_point='abides_gym.envs:SubGymMarketsMmRiskAvEnv_v1',
)


# REGISTER ENVS FOR RAY/RLLIB USE

register_env(
    "markets-daily_investor-v0",
    lambda config: SubGymMarketsDailyInvestorEnv_v0(**config),
)

register_env(
    "markets-execution-v0",
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)

register_env(
    "markets-mm-basic-v0",
    lambda config: SubGymMarketsMmBasicEnv_v0(**config),
)

register_env(
    "markets-mm-riskav-v0",
    lambda config: SubGymMarketsMmRiskAvEnv_v0(**config),
)