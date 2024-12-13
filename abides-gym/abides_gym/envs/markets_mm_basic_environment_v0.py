import importlib
from typing import Any, Dict, List
#from sklearn.preprocessing import MinMaxScaler

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv


class SubGymMarketsMmBasicEnv_v0(AbidesGymMarketsEnv):
    """
   Market Maker Basic v0 environment. It defines one of the ABIDES-Gym-markets environments.
   This environment presents a very basic example of a market maker agent where the agent tries to profit from the bid-ask spread throughout a single day.
   The Market Maker starts the day with cash but no position, then repeatedly posts simultaneous bid and ask limit orders of fixed size to maximize its
   reward at the end of the day (i.e., Execution PnL and Inventory PnL).

   NB: This agent does not take into account risks connected with inventory.


    Arguments:
        - background_config: the handcrafted agents configuration used for the environment
        - mkt_close: time the market day ends
        - timestep_duration: how long between 2 wakes up of the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - order_fixed_size: size of the order placed by the experimental gym agent
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer 
        - first_interval: how long the simulation is run before the first wake up of the gym experimental agent
        - reward_mode: can use a dense or sparse reward formulation
        - done_ratio: ratio (mark2market_t/starting_cash) that defines when an episode is done (if agent has lost too much mark to market value) -> keep it as a barrier if the mm agent perform too bad
        - debug_mode: arguments to change the info dictionary (lighter version if performance is an issue)

Execution MM:
    - Action Space: 
        - LMT bid/ask quotes, asymmetric at distance 0-4 from the best bid/ask.
        - LMT bid/ask quotes, asymmetric at distance 0-9 from the best bid/ask.
        - LMT bid/ask quotes, asymmetric at distance 0-14 from the best bid/ask.
        - LMT bid/ask quotes, asymmetric at distance 4-0 from the best bid/ask.
        - LMT bid/ask quotes, symmetric at distance 4 from the best bid/ask.
        - LMT bid/ask quotes, asymmetric at distance 4-9 from the best bid/ask.
        - LMT bid/ask quotes, asymmetric at distance 4-14 from the best bid/ask.
        - LMT bid/ask quotes, asymmetric at distance 9-0 from the best bid/ask. 
        - LMT bid/ask quotes, asymmetric at distance 9-4 from the best bid/ask. 
        - LMT bid/ask quotes, symmetric at distance 9 from the best bid/ask. 
        - LMT bid/ask quotes, asymmetric at distance 9-14 from the best bid/ask. 
        - LMT bid/ask quotes, asymmetric at distance 14-0 from the best bid/ask. 
        - LMT bid/ask quotes, asymmetric at distance 14-4 from the best bid/ask. 
        - LMT bid/ask quotes, asymmetric at distance 14-9 from the best bid/ask. 
        - LMT bid/ask quotes, symmetric at distance 14 from the best bid/ask. 
        - MKT to reduce the inventory exposure by 25%
 
    1. Before posting new orders, all existing ones are deleted with a Cancellation order.
    2. distance unit = half of market spread.

    - State Space:
        - Holdings (Inventory)
        - Imbalance
        - Market Spread
        - Last two mid-prices
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    def __init__(
        self,
        background_config: str = "rmsc04",#"rmsc03", 
        mkt_close: str = "16:00:00",
        timestep_duration: str = "60s", 
        starting_cash: int = 10_000_000, 
        order_fixed_size: int = 500, 
        state_history_length: int = 4, 
        market_data_buffer_length: int = 5,
        first_interval: str = "00:05:00",
        reward_mode: str = "dense", #"sparse"  
        done_ratio: float = 0.3,
        debug_mode: bool = False,  
        background_config_extra_kvargs={},
    ) -> None:
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )  #
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)  
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)  
        self.starting_cash: int = starting_cash  
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.reward_mode: str = reward_mode
        self.done_ratio: float = done_ratio
        self.debug_mode: bool = debug_mode

        # EXTRA FOR THE PURPOSE
        self.mid_price: None 
        self.spread: None
        self.mid_price_hist: None
        self.action :int = None 
        self.imbalance:float = None
        self.holdings: int = 0 #None 
        self.old_cash: int = starting_cash
        self.m2m: float = float(starting_cash)
        self.best_bid: int = 0
        self.best_ask: int = 0
        self.price_hist = []
        self.mm_bid: int = 0  
        self.mm_ask: int = 0

        # for RSI/Volatility EMA computation
        self.SMMA_up_old: float = 0.0
        self.SMMA_dwn_old: float = 0.0
        self.EMA_old: float = 0.0
        self.EMA_up_old: float = 0.0
        self.EMA_dwn_old: float = 0.0
        self.alpha: float = 2/(30+1) 


        # marked_to_market limit to STOP the episode  
        self.down_done_condition: float = self.done_ratio * starting_cash

        # CHECK PROPERTIES
        assert background_config in [
            "rmsc03",
            "rmsc04",
            "smc_01",
        ], "Select rmsc03, rmsc04 or smc_01 as config"

        assert (self.first_interval <= str_to_ns("16:00:00")) & (
            self.first_interval >= str_to_ns("00:00:00")
        ), "Select authorized FIRST_INTERVAL delay"

        assert (self.mkt_close <= str_to_ns("16:00:00")) & (
            self.mkt_close >= str_to_ns("09:30:00")
        ), "Select authorized market hours"

        assert reward_mode in [
            "sparse",
            "dense",
        ], "reward_mode needs to be dense or sparse"

        assert (self.timestep_duration <= str_to_ns("06:30:00")) & (
            self.timestep_duration >= str_to_ns("00:00:00")
        ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
            self.starting_cash >= 0
        ), "Select positive integer value for starting_cash"

        assert (type(self.order_fixed_size) == int) & (
            self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.state_history_length) == int) & (
            self.state_history_length >= 0
        ), "Select positive integer value for state_history_le"

        assert (type(self.market_data_buffer_length) == int) & (
            self.market_data_buffer_length >= 0
        ), "Select positive integer value for market_data_buffer_length"

        assert (
            (type(self.done_ratio) == float)
            & (self.done_ratio >= 0)
            & (self.done_ratio < 1)
        ), "Select positive float value for done_ratio between 0 and 1"

        assert debug_mode in [
            True,
            False,
        ], "reward_mode needs to be True or False"

        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
        )

        # Action Space: Only posting allowed
        self.num_actions: int = 16 #8 
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # State Space
        # [Inventory, Imbalance, Mkt_spread...] + Mid_price_vec[t-1,t] 
        # [Inventory, Imbalance_tot, Imbalance_5, Mkt_spread, Bid_exe_vol, Ask_exe_vol, time_to_end, direction feature] + Mid_price_vec[t-1,t] 
        self.num_state_features: int =  10 # + 4 2*(self.state_history_length - 1)
        # construct state space "box"
        self.state_highs: np.ndarray = np.array(
            [
                np.finfo(np.float32).max,  # Inventory
                1.0,  # Imbalance
                1.0,  # Imbalance 5
                np.finfo(np.float32).max,  # Mkt Spread
                np.finfo(np.float32).max,  # Bid executed vol 
                np.finfo(np.float32).max,  # Ask executed vol
                np.finfo(np.float32).max,  # Time to end 
                np.finfo(np.float32).max,  # direction
            ]
            + 2*[np.finfo(np.float32).max],  # Mid_Price_hist -> for padded_return : (self.state_history_length - 1)*
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                np.finfo(np.float32).min,  # Inventory
                0.0,  # Imbalance
                0.0,  # Imbalance 5
                np.finfo(np.float32).min,  # Mkt Spread
                np.finfo(np.float32).min,  # Bid executed vol 
                np.finfo(np.float32).min,  # Ask executed vol
                np.finfo(np.float32).min,  # Time to end
                np.finfo(np.float32).min,  # Direction
            ]
            +2* [np.finfo(np.float32).min],  # Mid_Price_hist
        ).reshape(self.num_state_features, 1)

        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32,
        )

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        """
        utility function that maps open ai action definition (integers) to environment API action definition (list of dictionaries)
        The action space ranges from 0 to 15 where:
        - `0`: Place the bid at the best_bid and the ask 4 ticks away from the best_ask.
        - `1`: Place the bid at the best_bid and the ask 9 ticks away from the best_ask.
        - `2`: Place the bid at the best_bid and the ask 14 ticks away from the best_ask.
        - `3`: Place the bid 4 ticks away from the best_bid and the ask at the best_ask.
        - `4`: Place the bid at the best_bid and the ask 14 ticks away from the best_ask.
        - `5`: Place the bid 4 ticks away from the best_bid and the ask 9 ticks away from the best_ask.
        - `6`: Place orders symmetrically 4 ticks away from the best_bid and best_ask.
        - `7`: Place the bid 9 ticks away from the best_bid and the ask at the best_ask.
        - `8`: Place the bid 9 ticks away from the best_bid and the ask 4 ticks away from the best_ask.
        - `9`: Place the bid at the best_bid and the ask 14 ticks away from the best_ask.
        - `10`: Place the bid 9 ticks away from the best_bid and the ask 14 ticks away from the best_ask.
        - `11`: Place the bid 14 ticks away from the best_bid and the ask at the best_ask.
        - `12`: Place the bid 14 ticks away from the best_bid and the ask 4 ticks away from the best_ask.
        - `13`: Place the bid 14 ticks away from the best_bid and the ask 9 ticks away from the best_ask.
        - `14`: Place orders symmetrically 14 ticks away from the best_bid and best_ask.
        - `15`: Place a Market Order to reduce the inventory exposure by 25%.


        Arguments:
            - action: integer representation of the different actions

        Returns:
            - action_list: list of the corresponding series of action mapped into abides env apis
        """
        if action == 0:
            self.action = 0
            self.mm_bid = self.best_bid
            self.mm_ask = self.best_ask + 4
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 4))}]
        elif action == 1:
            self.action = 1
            self.mm_bid = self.best_bid
            self.mm_ask = self.best_ask + 9
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 9))}]
        elif action == 2:
            self.action = 2
            self.mm_bid = self.best_bid
            self.mm_ask = self.best_ask + 14
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid ))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 14))}]
        elif action == 3:
            self.action = 3
            self.mm_bid = self.best_bid -4
            self.mm_ask = self.best_ask 
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 4))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask ))}]
        elif action == 4:
            self.action = 4
            self.mm_bid = self.best_bid -4
            self.mm_ask = self.best_ask +4
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 4))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 4))}]
        elif action == 5:
            self.action = 5
            self.mm_bid = self.best_bid -4
            self.mm_ask = self.best_ask +9
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 4))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 9))}]
        elif action == 6:
            self.action = 6
            self.mm_bid = self.best_bid -4
            self.mm_ask = self.best_ask +14
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 4))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 14))}]
        elif action == 7:
            self.action = 7
            self.mm_bid = self.best_bid -9
            self.mm_ask = self.best_ask 
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 9))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask ))}]
        elif action == 8:
            self.action = 8
            self.mm_bid = self.best_bid -9
            self.mm_ask = self.best_ask +4
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 9))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 4))}]
        elif action == 9:
            self.action = 9
            self.mm_bid = self.best_bid -9
            self.mm_ask = self.best_ask +9
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 9))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 9))}]
        elif action == 10:
            self.action = 10
            self.mm_bid = self.best_bid -9
            self.mm_ask = self.best_ask +14
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 9))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 14))}]
        elif action == 11:
            self.action = 11
            self.mm_bid = self.best_bid -14
            self.mm_ask = self.best_ask 
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 14))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 0))}]
        elif action == 12:
            self.action = 12
            self.mm_bid = self.best_bid -14
            self.mm_ask = self.best_ask +4
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 14))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 4))}]
        elif action == 13:
            self.action = 13
            self.mm_bid = self.best_bid -14
            self.mm_ask = self.best_ask +9
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 14))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 9))}] 
        elif action == 14:
            self.action = 14
            self.mm_bid = self.best_bid -14
            self.mm_ask = self.best_ask +14
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_bid - 14))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.best_ask + 14))}]
        elif action == 15:
            self.action = 15
            if int(np.abs(0.25*self.holdings)) != 0:
                return [{"type": "CCL_ALL"},
                       {"type": "MKT", "direction": "SELL" if (self.holdings) > 0  else "BUY",
                        "size": int(np.abs(0.25*self.holdings))
                      }]
            else:
                return [{"type": "CCL_ALL"}]  
        else:
            raise ValueError(f"Action {action} is not part of the actions supported by the function.")

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        method that transforms a raw state into a state representation

        Arguments:
            - raw_state: dictionary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state representation defining the MDP for the market maker basic v0 environment
        """

        """
        print("Stampa dati raw state")
        print("Bid Volume:", raw_state["parsed_volume_data"]["bid_volume"])
        print("Ask Volume:", raw_state["parsed_volume_data"]["ask_volume"])
        print("Total Volume:", raw_state["parsed_volume_data"]["total_volume"])
        print("Last_transaction Volume:", raw_state["parsed_volume_data"]["last_transaction"]) 
        print("Last_transaction Market:", raw_state["parsed_mkt_data"]["last_transaction"])
        print("bids:",raw_state["parsed_mkt_data"]["bids"])
        print("asks:",raw_state["parsed_mkt_data"]["asks"])
        """
        # 0)  Preliminary
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        # 1) Holdings/Inventory
        holdings = raw_state["internal_data"]["holdings"]
        self.holdings = holdings[-1]

        # 2) Imbalances
        imbalances = [
            markets_agent_utils.get_imbalance(b, a, depth=None)
            for (b, a) in zip(bids, asks)
        ]

        imbalances_5 = [
            markets_agent_utils.get_imbalance(b, a, depth=5)
            for (b, a) in zip(bids, asks)
        ]
        
        imbalances_3 = [
            markets_agent_utils.get_imbalance(b, a, depth=3)
            for (b, a) in zip(bids, asks)
        ]

        # 3) Mid prices
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        self.price_hist.append(mid_prices[-1])
        self.mid_price = mid_prices[-1] 

        # 4) Spread
        best_bids = [
            bids[0][0] if len(bids) > 0 else mid
            for (bids, mid) in zip(bids, mid_prices)
        ]
        self.best_bid = best_bids[-1]
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]
        self.best_ask = best_asks[-1]
        spreads = np.array(best_asks) - np.array(best_bids)
        self.spread = spreads[-1]
        self.imbalance = imbalances[-1]

        # Performance metric
        cash = raw_state["internal_data"]["cash"]
        marked_to_market = cash[-1] + self.holdings * self.mid_price
        self.m2m = marked_to_market

        # 5) Execution Volume Data & Book Volume Data
        exe_bid_vol = raw_state["parsed_volume_data"]["bid_volume"]
        exe_ask_vol = raw_state["parsed_volume_data"]["ask_volume"]
        exe_tot_vol = raw_state["parsed_volume_data"]["total_volume"]
        
        [bid_lob_vol, ask_lob_vol] = markets_agent_utils.get_volumes(bids[-1], asks[-1])

        # 6) Time state 
        current_time = raw_state["internal_data"]["current_time"][-1]
        time_to_end = (raw_state["internal_data"]["mkt_close"][-1] - current_time) / (raw_state["internal_data"]["mkt_close"][-1]-raw_state["internal_data"]["mkt_open"][-1])
        
        # 7) direction feature
        direction_features = np.array(mid_prices) - np.array(last_transactions)
        direction_feature = direction_features[-1]

        # Computation for additional features (beyond the benchmark settings)
        # Extenson 1: Cash
        cash = raw_state["internal_data"]["cash"]
        cash_tot_return = (cash[-1]-self.starting_cash)/self.starting_cash
        log_cash =  np.log(1+ cash[-1] / self.starting_cash) if cash[-1] > 0 else 0
        cash_step_return = (cash[-1]-self.old_cash)/self.old_cash

        # Extenson 2: Historical Data
        # 2.1) Price Volatility  
        price_vol = 0.0
        if len(self.price_hist)>1 and len(self.price_hist)<30:
            # possible alternative with moving average for early steps
            #MA_t = np.mean(self.price_hist[1:]) 
            #MA_t_1 = np.mean(self.price_hist[:-1])
            #price_vol = ((MA_t - MA_t_1) / MA_t_1) if MA_t_1 != 0 else 0.0
            EMA_t = self.price_hist[-1]*self.alpha + np.mean(self.price_hist)*(100-self.alpha)
            price_vol = ((EMA_t - self.EMA_old) / self.EMA_old) if self.EMA_old != 0 else 0.0
            self.EMA_old = EMA_t
        elif len(self.price_hist)>=30:
            EMA_t = self.price_hist[-1]*self.alpha + np.mean(self.price_hist[-30:])*(100-self.alpha)
            price_vol = (EMA_t - self.EMA_old) / self.EMA_old
            self.EMA_old = EMA_t
            #price_vol = np.std(self.price_hist[-30:])
            
        # 2.2) Price return 
        returns = np.diff(mid_prices)
        padded_returns = np.zeros(self.state_history_length - 1)
        padded_returns[-len(returns) :] = (
            returns if len(returns) > 0 else padded_returns
        )

        # 2.3) Mid Price Hist 
        if len(mid_prices) >= 2:
            mid_price_hist =  np.array(mid_prices[-2:])
            if self.mid_price_hist[0] != 0:
                #mid_price_return = np.diff(self.mid_price_hist)[0] / self.mid_price_hist[0]
                mid_price_return = np.log(mid_prices[-1] / mid_prices[-2])
            else:
                mid_price_return = 0
        elif len(mid_prices) == 1:
            mid_price_hist = np.array([0, mid_prices[0]])
            mid_price_return = 0
        else:
            mid_price_hist = np.zeros(2)
            mid_price_return = 0

        self.mid_price_hist = mid_price_hist
        self.mid_price = mid_price_hist[-1]
        
        # Extension 3: Book Data 
        # 3.1) Sides Depth:
        bid_depth = len(bids[-1])
        ask_depth = len(asks[-1])

        # 3.2) Executed Volumes Variation:
        exe_bid_diff = np.diff(exe_bid_vol)
        exe_ask_diff = np.diff(exe_ask_vol)
        print("exe_bid_vol:", exe_bid_vol)
        print("exe_bid_vol[-1]:", exe_bid_vol[-1])

        exe_bid_vol_variation = np.zeros(self.state_history_length - 1)
        exe_ask_vol_variation = np.zeros(self.state_history_length - 1)
        
        exe_bid_vol_variation [-len(exe_bid_diff) :] = (
            exe_bid_diff if len(exe_bid_diff)>0 else exe_bid_vol_variation
        )
        exe_ask_vol_variation [-len(exe_ask_diff) :] = (
            exe_ask_diff if len(exe_ask_diff)>0 else exe_ask_vol_variation
        )
        """
        exe_bid_vol_variation = np.diff(bid_vol)[-1] if len(bid_vol)>1 else 0
        exe_ask_vol_variation = np.diff(ask_vol)[-1] if len(ask_vol)>1 else 0
        """

        # 3.3) Side Differential Price
        # Controllo best_bids e bids
        if best_bids and bids and len(bids[-1]) > 0 and len(bids[-1][-1]) > 0:
            delta_bid = best_bids[-1] - bids[-1][-1][0]
        else:
            delta_bid = 0  
        
        if best_asks and asks and len(asks[-1]) > 0 and len(asks[-1][-1]) > 0:
           delta_ask = asks[-1][-1][0] - best_asks[-1]
        else:
           delta_ask = 0 

        # 3.4) Difference between the market maker’s bid/ask and the best bid/ask
        if self.action != None:
            delta_mm_bid = (self.best_bid - self.mm_bid) if self.action != 15 else 1000
            delta_mm_ask = (self.mm_bid - self.best_ask) if self.action != 15 else 1000
        else:
            delta_mm_ask = 0
            delta_mm_bid = 0

        # Extension 4: On Volume details
        # 4.1) Market Maker Executed Volume
        inter_wakeup_executed_orders = raw_state["internal_data"]["inter_wakeup_executed_orders"][-1]
        exe_qt_ask = 0
        exe_qt_bid = 0
        for order in inter_wakeup_executed_orders:
            if order.side == "ASK":
                exe_qt_ask += order.quantity  
            else:
                 exe_qt_bid += order.quantity
        #exe_qt_ask /= self.order_fixed_size
        #exe_qt_bid /= self.order_fixed_size
        exe_MM = (exe_qt_ask+exe_qt_bid)/(2*self.order_fixed_size)

        # 4.2) MM exe volume over the total executed volume
        exe_pct_tot_volume = exe_MM / raw_state["parsed_volume_data"]["total_volume"][-1]

        # Extension 5: Technical Index
        # RSI (Relative Strenght Index)

        # 5.1) RSI with moving average
        rsi_MA = markets_agent_utils.rsi_index(mid_prices)
        
        """
        # 5.2) RSI 2: with SMMA
        retruns = np.diff(mid_prices)
        if len(returns)>1:
            if returns[-1]>0:
              SMMA_up = (self.SMMA_up_old*(len(returns)-1) + returns[-1])/len(returns)
              self.SMMA_up_old = SMMA_up
              SMMA_dwn = self.SMMA_dwn_old
            elif returns[-1]<0:
              SMMA_dwn = (self.SMMA_dwn_old*(len(returns)-1) + abs(returns[-1]))/len(returns)
              self.SMMA_dwn_old = SMMA_dwn 
              SMMA_up = self.SMMA_up_old
            else:
                SMMA_up = self.SMMA_up_old*(len(returns)-1)/len(returns)
                SMMA_dwn = self.SMMA_dwn_old*(len(returns)-1)/len(returns)

            if SMMA_dwn != 0:  
                rs = SMMA_up / SMMA_dwn
                rsi_SMMA = 100 - 100/(1+rs)
            else:
                rsi = 100.0
        elif len(returns) == 1: 
            if returns[0]>0:       
                self.SMMA_up_old = returns[0]
                # self.SMMA_dwn_old = 0.0
                rsi_SMMA = 100.0
            else:
                self.SMMA_dwn_old = abs(returns[0])
                #self.SMMA_up_old = 0.0
                rsi_SMMA = 0.0
        else:
            rsi_SMMA = 50.0
        """
        # 5.3) RSI 3: with EMA 
        if len(self.price_hist) <= 1:
            rsi_EMA = 50.0
        else:
            ret = np.diff(self.price_hist) 
            if  len(ret)<=30:
               # first 30 steps are evaluated with Moving Average
               MA_up = np.mean(np.where(ret>0, ret, 0))              
               MA_dwn= np.mean(np.where(ret<0, -ret, 0))
               rsi = (100 - 100/(1 + MA_up/MA_dwn)) if MA_dwn != 0 else 100.0
               if len(ret) == 30:
                self.EMA_up_old = MA_up
                self.EMA_dwn_old = MA_dwn
            elif  len(ret)> 30:
              # with more than 30 observation exponetial moving average is used
              if ret[-1] >= 0:
                EMA_up = self.EMA_up_old*(1-self.alpha) + ret[-1]*self.alpha
                self.EMA_up_old = EMA_up
                EMA_dwn = self.EMA_dwn_old*(1-self.alpha)
                self.EMA_dwn_old = EMA_dwn
                rsi_EMA = (100 - 100/(1 + EMA_up/EMA_dwn)) if EMA_dwn != 0 else 100.0
              else:
                EMA_up = self.EMA_up_old*(1-self.alpha)
                self.EMA_up_old = EMA_up
                EMA_dwn = self.EMA_dwn_old*(1-self.alpha) + ret[-1]*self.alpha
                self.EMA_dwn_old = EMA_dwn
                rsi_EMA = (100 - 100/(1 + EMA_up/EMA_dwn)) if EMA_dwn != 0 else 100.0
        #"""
        
        # EXTRA: On Detailed LOB Data
        lob_data = np.zeros(20)
        levels = 5
        for it, (price_lev, vol) in enumerate(bids[-1]):
            if it == levels:
                break
            lob_data[2*it] = price_lev #(price_lev/self.mid_price)-1 # price_lev
            lob_data[2*it + 1] = vol
        for it, (price_lev, vol) in enumerate(asks[-1]):
            if it == levels:
                break
            lob_data[2*it+10] = price_vol #(price_lev/self.mid_price)-1 # price_lev
            lob_data[(2*it+1)+10] = vol

        
        #NORMALIZATION (via calibrated const)    

        #MinMax 
        MinMax_Holdings = (holdings[-1] - (-7749))/(7155-(-7749))
        Min_Max_bid_vol = (exe_bid_vol[-1]-17)/(4078-17)
        Min_Max_ask_vol = (exe_ask_vol[-1]-34)/(3358-34)
        Min_Max_mid_pice_hist = (mid_price_hist - 1000459)/(100459 - 99445.5)

        #z-score
        z_score_holdings = (holdings[-1] - (-450))/1747
        z_score_bid_vol = (exe_bid_vol[-1]-1346.64)/ 601.64
        z_score_ask_vol = (exe_ask_vol[-1]-1324.18)/ 590.409
        z_score_mid_price_hist = (mid_price_hist - 99999.3843)/154.6205
        z_score_spread = (spreads[-1]- 1.69)/2.48

        """
        #NEW NORMAILZATION CONSTANT 
        z_score_holdings = (holdings[-1] - (-1709.38))/ 2554.88
        z_score_bid_vol = (exe_bid_vol-1364.25)/608.12
        z_score_ask_vol = (exe_ask_vol-1359.88)/ 604.16
        z_score_mid_price_hist = (mid_price_hist - 99998.50)/186.307
        z_score_spread = (spreads[-1]- 1.6088)/1.987
        """
        clipped_spread = spreads[-1] /  5 if np.abs(spreads[-1]) < 6 else 1
        
        
        # 9) Compute State (Holdings, Imbalance, Spread, Mid_price_vec[t-1,t])
        computed_state = np.array(
            [holdings[-1], imbalances[-1], imbalances_5[-1], spreads[-1], exe_bid_vol[-1], exe_ask_vol[-1], time_to_end, direction_feature]
            #+ lob_data.tolist()
            #+ exe_bid_vol_variation.tolist()
            #+ exe_ask_vol_variation.tolist()
            + mid_price_hist.tolist(),
            #+ mid_price_hist.tolist(),
            dtype=np.float32,
        )
        """
        computed_state = np.array(
            [z_score_holdings, imbalances[-1], imbalances_5[-1], clipped_spread, z_score_bid_vol, z_score_ask_vol, time_to_end, direction_feature, log_cash]
            + z_score_mid_price_hist.tolist(),
            dtype=np.float32,
        )
        
        computed_state = np.array(
            [MinMax_Holdings, imbalances[-1], imbalances_5[-1], clipped_spread, Min_Max_bid_vol, Min_Max_ask_vol, time_to_end, direction_feature]
            + Min_Max_mid_pice_hist.tolist(),
            dtype=np.float32,
        )
        """
        #print("STATE SPACE:", computed_state)
        return computed_state.reshape(self.num_state_features, 1)

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the reward obtained during the step

        Arguments:
            - raw_state: dictionary that contains raw simulation information obtained from the gym experimental agent

        Returns:mkt
            - reward: immediate reward computed at each step  for the market maker basic v0 environment
        """
        if self.reward_mode == "dense":
            # Dense Reward here
            # Agents get reward after each action

            # 0)  Preliminary
            bids = raw_state["parsed_mkt_data"]["bids"]
            asks = raw_state["parsed_mkt_data"]["asks"]
            last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

            # 1) Holdings\Inventory
            holdings = raw_state["internal_data"]["holdings"]
            
            # 2) Available Cash
            cash = raw_state["internal_data"]["cash"]
        
            # 3) Inventory PnL
            Inventory_PnL =  holdings * np.diff(self.mid_price_hist)[0] 

            # 4) Execution PnL 
            inter_wakeup_executed_orders = raw_state["internal_data"][
            "inter_wakeup_executed_orders"]
            
            if len(inter_wakeup_executed_orders) == 0:
                Execution_PnL = 0
            else:
                
                Execution_PnL = sum(
                        (self.mid_price - order.fill_price) * order.quantity
                         if order.side == "BID" else (order.fill_price - self.mid_price) * order.quantity
                            for order in inter_wakeup_executed_orders
                        )
             # 7) Time state 
            current_time = raw_state["internal_data"]["current_time"]
            time_to_end = (raw_state["internal_data"]["mkt_close"] - current_time) / (raw_state["internal_data"]["mkt_close"]-raw_state["internal_data"]["mkt_open"])
        
            # 5) Reward
            reward =  Execution_PnL +  Inventory_PnL # - np.maximum(0,0.6*Inventory_PnL) #*1e2 *np.exp(1-time_to_end)
            self.old_cash = cash
            print("Execution_PnL:", Execution_PnL)
            print("Inventory_PnL:", Inventory_PnL)
            

            # 6) Order Size Normalization of Reward
            reward = reward / (2*self.order_fixed_size)

            # 7) Time Normalization of Reward
            num_ns_day = (16 - 9.5) * 60 * 60 * 1e9
            step_length = self.timestep_duration
            num_steps_per_episode = num_ns_day / step_length
            reward = reward / num_steps_per_episode
            
            return reward

        elif self.reward_mode == "sparse":
            return 0

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the final step reward update (if needed)

        Arguments:
            - raw_state: dictionary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode for the market maker basic v0 environment
        """
        if self.reward_mode == "dense":
            return 0

        elif self.reward_mode == "sparse":
            # Sparse Reward here
            # Agents get reward at the end of the episode
            # reward is computed for the last step for each episode
            # can update with additional reward at end of episode depending on scenario (Not in this case)

            # 0) Preliminary
            bids = raw_state["parsed_mkt_data"]["bids"]
            asks = raw_state["parsed_mkt_data"]["asks"]
            last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

            # 1) Holdings\Inventory
            holdings = raw_state["internal_data"]["holdings"]

            # 2) Available Cash
            cash = raw_state["internal_data"]["cash"]

            # 3) Inventory PnL 
            #Inventory_PnL =  holdings * np.diff(self.mid_price_hist)[0] 
            Inventory_PnL = holdings * self.mid_price_hist[-1]

            # 4) (Cumulative)Execution PnL 
            episode_executed_orders = raw_state["internal_data"][
            "episode_executed_orders"]
            
            if len(episode_executed_orders) == 0:
                Execution_PnL = 0
            else:
                Execution_PnL = sum(
                         (order.fill_price * order.quantity) * (-1 if order.side == "BID" else 1)
                            for order in episode_executed_orders
                        )
                """
                Execution_PnL = sum(
                        (self.mid_price - order.fill_price) * order.quantity
                         if order.side == "BID" else (order.fill_price - self.mid_price) * order.quantity
                            for order in episode_executed_orders
                        )
                """
            # 5) Reward
            reward = Execution_PnL +  Inventory_PnL #) #*1e2
            #print("Execution_PnL_final:", Execution_PnL)
            #print("Inventory_PnL_final:", Inventory_PnL)

            # 6) Order Size Normalization of Reward
            reward = reward / (2*self.order_fixed_size)

            # 7) Time Normalization of Reward
            num_ns_day = (16 - 9.5) * 60 * 60 * 1e9
            step_length = self.timestep_duration
            num_steps_per_episode = num_ns_day / step_length
            reward = reward / num_steps_per_episode
            reward = (holdings* self.mid_price + cash - self.starting_cash)
            return reward


    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """
        method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not  for the market maker basic v0 environment
        """
        # episode can stop because market closes or because some condition is met
        # here choose to make it trader has lost too much money

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 2) Available Cash
        cash = raw_state["internal_data"]["cash"]

        # 3) Last Known Market Transaction Price
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        # 4) compute the marked to market
        marked_to_market = cash + holdings * self.mid_price

        # 5) comparison
        done = marked_to_market <= self.down_done_condition
        done = False
        return done

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        method that transforms a raw state into an info dictionary

        Arguments:
            - raw_state: dictionary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionary computed at each step for the market maker basic v0 environment
        """
        # Agent cannot use this info for taking decision
        # only for debugging

        # 1) Last Known Market Transaction Price
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        # 2) Last Known best bid
        bids = raw_state["parsed_mkt_data"]["bids"]
        best_bid = bids[0][0] if len(bids) > 0 else last_transaction

        # 3) Last Known best ask
        asks = raw_state["parsed_mkt_data"]["asks"]
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction

        # 4) Available Cash
        cash = raw_state["internal_data"]["cash"]

        # 5) Current Time
        current_time = raw_state["internal_data"]["current_time"]

        # 6) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 7) Spread
        spread = best_ask - best_bid

        # 8) OrderBook features
        orderbook = {
            "asks": {"price": {}, "volume": {}},
            "bids": {"price": {}, "volume": {}},
        }

        for book, book_name in [(bids, "bids"), (asks, "asks")]:
            for level in [0, 1, 2]:
                price, volume = markets_agent_utils.get_val(bids, level)
                orderbook[book_name]["price"][level] = np.array([price]).reshape(-1)
                orderbook[book_name]["volume"][level] = np.array([volume]).reshape(-1)

        # 9) order_status
        order_status = raw_state["internal_data"]["order_status"]

        # 10) mkt_open
        mkt_open = raw_state["internal_data"]["mkt_open"]

        # 11) mkt_close
        mkt_close = raw_state["internal_data"]["mkt_close"]

        # 12) last vals
        last_bid = markets_agent_utils.get_last_val(bids, last_transaction)
        last_ask = markets_agent_utils.get_last_val(asks, last_transaction)

        # 13) spreads
        wide_spread = last_ask - last_bid
        ask_spread = last_ask - best_ask
        bid_spread = best_bid - last_bid

        # 14) Bid/Ask vol
        exe_bid_vol =  raw_state["parsed_volume_data"]["bid_volume"]
        exe_ask_vol =  raw_state["parsed_volume_data"]["ask_volume"]
        [bid_lob_vol, ask_lob_vol] = markets_agent_utils.get_volumes(bids, asks)     

        if self.debug_mode == True:
            return {
                "mid price": self.mid_price,
                "last_transaction": last_transaction,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": self.spread,
                "bids": bids,
                "asks": asks,
                "cash": cash,
                "current_time": current_time,
                "holdings": holdings,
                "orderbook": orderbook,
                "order_status": order_status,
                "mkt_open": mkt_open,
                "mkt_close": mkt_close,
                "last_bid": last_bid,
                "last_ask": last_ask,
                "wide_spread": wide_spread,
                "ask_spread": ask_spread,
                "bid_spread": bid_spread,
            }
        else:
            return {
                "holdings": holdings,
                "spread": self.spread,
                "Mid Price": self.mid_price,                
                "bid-vol": bid_lob_vol,
                "ask-vol": ask_lob_vol,
                "action": self.action,
                "book imbalance": self.imbalance,
                "value": self.m2m
            }

