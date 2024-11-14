import importlib
from typing import Any, Dict, List

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
        - LMT bid/ask quotes, symmetric at distance 1 from the mid-price
        - LMT bid/ask quotes, symmetric at distance 2 from the mid-price
        - LMT bid/ask quotes, symmetric at distance 3 from the mid-price
        - LMT bid/ask quotes, asymmetric at distance 1-3 from the mid-price
        - LMT bid/ask quotes, asymmetric at distance 3-1 from the mid-price
        - LMT bid/ask quotes, asymmetric at distance 2-5 from the mid-price
        - LMT bid/ask quotes, asymmetric at distance 5-2 from the mid-price
    NB:
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
        background_config: str = "rmsc04", #"rmsc03"
        mkt_close: str = "16:00:00",
        timestep_duration: str = "60s",
        starting_cash: int = 1_000_000, 
        order_fixed_size: int = 100, 
        state_history_length: int = 4,
        market_data_buffer_length: int = 5,
        first_interval: str = "00:05:00",
        reward_mode: str = "dense", #Or sparse  
        done_ratio: float = 0.3,
        debug_mode: bool = False,  
        background_config_extra_kvargs={},
    ) -> None:
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )  #
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)  #
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)  #
        self.starting_cash: int = starting_cash  #
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
        self.num_actions: int = 7 
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # State Space
        # [Inventory, Imbalance, Mkt_spread] + Mid_price_vec[t-1,t] 
        self.num_state_features: int = 5

        # construct state space "box"
        self.state_highs: np.ndarray = np.array(
            [
                np.finfo(np.float32).max,  # Inventory
                1.0,  # Imbalance
                np.finfo(np.float32).max,  # Mkt Spread
            ]
            + 2* [np.finfo(np.float32).max],  # Mid_Price_hist
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                np.finfo(np.float32).min,  # Inventory
                0.0,  # Imbalance
                np.finfo(np.float32).min,  # Mkt Spread
            ]
            + 2* [np.finfo(np.float32).min],  # Mid_Price_hist
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
        The action space ranges [0, 1, 2, 3, 4, 5, 6] where:
        - `0`: Place orders symmetric at distance 1 from the mid-price.
        - `1`: Place orders symmetric at distance 2 from the mid-price.
        - `2`: Place orders symmetric at distance 3 from the mid-price.
        - `3`: Place orders skewed with bid at distance 1 and ask at distance 3 from the mid-price.
        - `4`: Place orders skewed with bid at distance 3 and ask at distance 1 from the mid-price.
        - `5`: Place orders skewed with bid at distance 2 and ask at distance 5 from the mid-price.
        - `6`: Place orders skewed with bid at distance 5 and ask at distance 2 from the mid-price.

        Arguments:
            - action: integer representation of the different actions

        Returns:
            - action_list: list of the corresponding series of action mapped into abides env apis
        """
        if action == 0:
            self.action = 0
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + self.spread / 2))}]
        elif action == 1:
            self.action = 1
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - 2*self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + 2*self.spread / 2))}]
        elif action == 2:
            self.action = 2
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - 3*self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + 3*self.spread / 2))}]
        elif action == 3:
            self.action = 3
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + 3*self.spread / 2))}]
        elif action == 4:
            self.action = 4
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - 3*self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + self.spread / 2))}]
        elif action == 5:
            self.action = 5
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - 2*self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + 5*self.spread / 2))}]
        elif action == 6:
            self.action = 6
            return [{"type": "CCL_ALL"},
                    {"type": "LMT", "direction": "BUY", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price - 5*self.spread / 2))},
                    {"type": "LMT", "direction": "SELL", "size": self.order_fixed_size,
                    "limit_price": int(round(self.mid_price + 2*self.spread / 2))}]
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
        # 0)  Preliminary
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        # 1) Holdings/Inventory
        holdings = raw_state["internal_data"]["holdings"]

        # 2) Imbalance
        imbalances = [
            markets_agent_utils.get_imbalance(b, a, depth=5)
            for (b, a) in zip(bids, asks)
        ]

        # 3) MID PRICES 
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]

        if len(mid_prices) >= 2:
            mid_price_hist = mid_prices[-2:]
        elif len(mid_prices) == 1:
            mid_price_hist = np.array([0, mid_prices[0]])
        else:
            mid_price_hist = np.zeros(2)
        self.mid_price_hist = mid_price_hist
        self.mid_price = mid_price_hist[-1]

        # 4) Spread
        best_bids = [
            bids[0][0] if len(bids) > 0 else mid
            for (bids, mid) in zip(bids, mid_prices)
        ]
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]
        spreads = np.array(best_asks) - np.array(best_bids)
        self.spread = spreads[-1]
        self.imbalance = imbalances[-1]

        # 5) Compute State (Holdings, Imbalance, Spread, Mid_price_vec[t-1,t])
        computed_state = np.array(
            [holdings[-1], imbalances[-1], spreads[-1]]
            + list(mid_price_hist),
            dtype=np.float32,
        )
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
            print("Old Mid-Price:", self.mid_price_hist[0])
            print("Mid-Price:", self.mid_price)
            print("Delta-Mid-Price:",np.diff(self.mid_price_hist)[0])
            print("Inventory:", holdings)
            #for order in inter_wakeup_executed_orders:
            #   print("order ID", order.id)
            #   print("SIDE:",order.side)
            #   print("fill Price:",order.fill_price)
            #   print("order Quantity:", order.quantity)
            
            if len(inter_wakeup_executed_orders) == 0:
                Execution_PnL = 0
            else:
                Execution_PnL = sum(
                        (self.mid_price - order.fill_price) * order.quantity
                         if order.side == "BID" else (order.fill_price - self.mid_price) * order.quantity
                            for order in inter_wakeup_executed_orders
                        )
            # 5) Reward
            reward = Execution_PnL + Inventory_PnL # + cash 
            #print("Execution_PnL:", Execution_PnL)
            #print("Inventory_PnL:", Inventory_PnL)
            #print("Cash:", cash)
            

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
            Inventory_PnL =  holdings * np.diff(self.mid_price_hist)[0] 

            # 4) (Cumulative)Execution PnL 
            episode_executed_orders = raw_state["internal_data"][
            "episode_executed_orders"]
            
            if len(episode_executed_orders) == 0:
                Execution_PnL = 0
            else:
                Execution_PnL = sum(
                        (self.mid_price - order.fill_price) * order.quantity
                         if order.side == "BID" else (order.fill_price - self.mid_price) * order.quantity
                            for order in episode_executed_orders
                        )
            # 5) Reward
            reward = Execution_PnL + Inventory_PnL 
            #print("Execution_PnL:", Execution_PnL)
            #print("Inventory_PnL:", Inventory_PnL)

            # 6) Order Size Normalization of Reward
            reward = reward / (2*self.order_fixed_size)

            # 7) Time Normalization of Reward
            num_ns_day = (16 - 9.5) * 60 * 60 * 1e9
            step_length = self.timestep_duration
            num_steps_per_episode = num_ns_day / step_length
            reward = reward / num_steps_per_episode
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
        done = False # temporary
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

        

        if self.debug_mode == True:
            return {
                "mid price": self.mid_price,
                "last_transaction": last_transaction,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
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
                "spread": spread,
                "action": self.action,
                "book imbalance": self.imbalance
            }
