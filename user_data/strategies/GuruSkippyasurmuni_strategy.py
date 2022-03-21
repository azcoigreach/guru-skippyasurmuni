# Guru Skippyasurmuni Strategy
# Author: AZcoigreach
# github: https://github.com/azcoigreach/

#             ___  _   _  ___  _   _                                                 
#            / __|| | | || _ \| | | |                                                
#           | (_ || |_| ||   /| |_| |                                                
#            \___| \___/ |_|_\ \___/                                                 
#    ___  _  __ ___  ___  ___ __   __ _    ___  _   _  ___  __  __  _   _  _  _  ___ 
#   / __|| |/ /|_ _|| _ \| _ \\ \ / //_\  / __|| | | || _ \|  \/  || | | || \| ||_ _|
#   \__ \| ' <  | | |  _/|  _/ \ V // _ \ \__ \| |_| ||   /| |\/| || |_| || .` | | | 
#   |___/|_|\_\|___||_|  |_|    |_|/_/ \_\|___/ \___/ |_|_\|_|  |_| \___/ |_|\_||___|
#        

'''
Guru Skippyasurmuni - 
Even though Skippy is designed to be easily modified and hyperopted. However, Skippy has alread spent a lot of time thinking about the best way to trade crypto.
He has come to the conclusing - simple is better.  A couple basic indicators and some simple rules will yield the best results in every condition.

'''

# Hyperopting
'''
docker-compose run --rm freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --strategy GuruSkippyasurmuni -e 2000 --timerange=20220314- --eps
'''

# Backtesting - Powershell
'''
 $1 = 20220314 ; $2 = "day" ;  
 docker-compose run --rm freqtrade backtesting --datadir user_data/data/binanceus --config /freqtrade/user_data/SkippyGod_config.json --export trades  -s SkippyGod --fee 0.00075 --timerange=$1- --breakdown $2 --eps ; 
 docker-compose run --rm freqtrade plot-dataframe -s SkippyGod --config /freqtrade/user_data/SkippyGod_config.json -i 5m --timerange=$1- --indicators2 AROONOSC-5 RSI-5  ; 
 docker-compose run --rm freqtrade plot-profit -s SkippyGod --config /freqtrade/user_data/SkippyGod_config.json -i 5m --timerange=$1-
'''

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from random import shuffle

# Initiate genes splicing --> Source: MaBlue GodStrNew strategy --> Target: Guru Skippyasurmuni strategy.

god_genes = set()
########################### SETTINGS ##############################
# RSI and an Aroon Oscillator are the only two metrics you need to catch the highs and lows

god_genes = {
    'RSI',                  # Relative Strength Index
    'AROONOSC',             # Aroon Oscillator
   }

timeperiods = [5, 6, 12, 15, 50, 55, 100, 110]
operators = [
    "D",  # Disabled gene
    ">",  # Indicator, bigger than cross indicator
    "<",  # Indicator, smaller than cross indicator
    "=",  # Indicator, equal with cross indicator
    "C",  # Indicator, crossed the cross indicator
    "CA",  # Indicator, crossed above the cross indicator
    "CB",  # Indicator, crossed below the cross indicator
    ">R",  # Normalized indicator, bigger than real number
    "=R",  # Normalized indicator, equal with real number
    "<R",  # Normalized indicator, smaller than real number
    "/>R",  # Normalized indicator devided to cross indicator, bigger than real number
    "/=R",  # Normalized indicator devided to cross indicator, equal with real number
    "/<R",  # Normalized indicator devided to cross indicator, smaller than real number
]

# number of candles to check up,don,off trend.
TREND_CHECK_CANDLES = 4
DECIMALS = 3

########################### END SETTINGS ##########################
# DATAFRAME = DataFrame()

god_genes = list(god_genes)
# print('selected indicators for optimzatin: \n', god_genes)

god_genes_with_timeperiod = list()
for god_gene in god_genes:
    for timeperiod in timeperiods:
        god_genes_with_timeperiod.append(f'{god_gene}-{timeperiod}')

# Let give somethings to CatagoricalParam to Play with them
# When just one thing is inside catagorical lists
# TODO: its Not True Way :) <-- But it's good enough
if len(god_genes) == 1:
    god_genes = god_genes*2
if len(timeperiods) == 1:
    timeperiods = timeperiods*2
if len(operators) == 1:
    operators = operators*2


def normalize(df):
    df = (df-df.min())/(df.max()-df.min())
    return df


def gene_calculator(dataframe, indicator):
    # Cuz Timeperiods not effect calculating CDL patterns recognations
    if 'CDL' in indicator:
        splited_indicator = indicator.split('-')
        splited_indicator[1] = "0"
        new_indicator = "-".join(splited_indicator)
        # print(indicator, new_indicator)
        indicator = new_indicator

    gene = indicator.split("-")

    gene_name = gene[0]
    gene_len = len(gene)

    if indicator in dataframe.keys():
        # print(f"{indicator}, calculated befoure")
        # print(len(dataframe.keys()))
        return dataframe[indicator]
    else:
        result = None
        # For Pattern Recognations
        if gene_len == 1:
            # print('gene_len == 1\t', indicator)
            result = getattr(ta, gene_name)(
                dataframe
            )
            return normalize(result)
        elif gene_len == 2:
            # print('gene_len == 2\t', indicator)
            gene_timeperiod = int(gene[1])
            result = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            )
            return normalize(result)
        # For
        elif gene_len == 3:
            # print('gene_len == 3\t', indicator)
            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            result = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            ).iloc[:, gene_index]
            return normalize(result)
        # For trend operators(MA-5-SMA-4)
        elif gene_len == 4:
            # print('gene_len == 4\t', indicator)
            gene_timeperiod = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            )
            return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))
        # For trend operators(STOCH-0-4-SMA-4)
        elif gene_len == 5:
            # print('gene_len == 5\t', indicator)
            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_index}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            ).iloc[:, gene_index]
            return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))


def condition_generator(dataframe, operator, indicator, crossed_indicator, real_num):

    condition = (dataframe['volume'] > 10)

    # TODO : it ill callculated in populate indicators.

    dataframe[indicator] = gene_calculator(dataframe, indicator)
    dataframe[crossed_indicator] = gene_calculator(dataframe, crossed_indicator)

    indicator_trend_sma = f"{indicator}-SMA-{TREND_CHECK_CANDLES}"
    if operator in ["UT", "DT", "OT", "CUT", "CDT", "COT"]:
        dataframe[indicator_trend_sma] = gene_calculator(dataframe, indicator_trend_sma)

    if operator == ">":
        condition = (
            dataframe[indicator] > dataframe[crossed_indicator]
        )
    elif operator == "=":
        condition = (
            np.isclose(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "<":
        condition = (
            dataframe[indicator] < dataframe[crossed_indicator]
        )
    elif operator == "C":
        condition = (
            (qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])) |
            (qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator]))
        )
    elif operator == "CA":
        condition = (
            qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "CB":
        condition = (
            qtpylib.crossed_below(
                dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == ">R":
        condition = (
            dataframe[indicator] > real_num
        )
    elif operator == "=R":
        condition = (
            np.isclose(dataframe[indicator], real_num)
        )
    elif operator == "<R":
        condition = (
            dataframe[indicator] < real_num
        )
    elif operator == "/>R":
        condition = (
            dataframe[indicator].div(dataframe[crossed_indicator]) > real_num
        )
    elif operator == "/=R":
        condition = (
            np.isclose(dataframe[indicator].div(dataframe[crossed_indicator]), real_num)
        )
    elif operator == "/<R":
        condition = (
            dataframe[indicator].div(dataframe[crossed_indicator]) < real_num
        )
    elif operator == "UT":
        condition = (
            dataframe[indicator] > dataframe[indicator_trend_sma]
        )
    elif operator == "DT":
        condition = (
            dataframe[indicator] < dataframe[indicator_trend_sma]
        )
    elif operator == "OT":
        condition = (

            np.isclose(dataframe[indicator], dataframe[indicator_trend_sma])
        )
    elif operator == "CUT":
        condition = (
            (
                qtpylib.crossed_above(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            ) &
            (
                dataframe[indicator] > dataframe[indicator_trend_sma]
            )
        )
    elif operator == "CDT":
        condition = (
            (
                qtpylib.crossed_below(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            ) &
            (
                dataframe[indicator] < dataframe[indicator_trend_sma]
            )
        )
    elif operator == "COT":
        condition = (
            (
                (
                    qtpylib.crossed_below(
                        dataframe[indicator],
                        dataframe[indicator_trend_sma]
                    )
                ) |
                (
                    qtpylib.crossed_above(
                        dataframe[indicator],
                        dataframe[indicator_trend_sma]
                    )
                )
            ) &
            (
                np.isclose(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            )
        )

    return condition, dataframe

class GuruSkippyasurmuni(IStrategy):
    INTERFACE_VERSION = 2

    # Variables
    DATESTAMP = 0    
    COUNT = 0
    custom_info = { }

    # Skippy's BUY and SELL conditions
    # Buy hyperspace params:
    buy_params = {
        "buy_crossed_indicator0": "STOCHRSI-0-15",
        "buy_crossed_indicator1": "MACDFIX-0-15",
        "buy_crossed_indicator2": "STOCHRSI-0-55",
        "buy_indicator0": "MACDFIX-0-15",
        "buy_indicator1": "RSI-5",
        "buy_indicator2": "AROONOSC-5",
        "buy_operator0": "D",
        "buy_operator1": "<R",
        "buy_operator2": "=R",
        "buy_real_num0": 0.35,
        "buy_real_num1": 0.30,
        "buy_real_num2": 0,
    }

        # Sell hyperspace params:
    sell_params = {
        "sell_crossed_indicator0": "STOCHRSI-0-15",
        "sell_crossed_indicator1": "MACDFIX-0-15",
        "sell_crossed_indicator2": "STOCHRSI-0-15",
        "sell_indicator0": "MACDFIX-0-15",
        "sell_indicator1": "RSI-5",
        "sell_indicator2": "AROONOSC-5",
        "sell_operator0": "D",
        "sell_operator1": ">R",
        "sell_operator2": "=R",
        "sell_real_num0": 0.65,
        "sell_real_num1": 0.9,
        "sell_real_num2": 1.0,
    }    

    # ROI table:
    minimal_roi = {
        "0": 1000.0
    }

    # Stoploss:
    stoploss = -1.0

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    # timeframe = '1m'
    process_only_new_candles = True

    # DCA config
    position_adjustment_enable = True
    max_entry_position_adjustment = 24
    dca_multiplier = 0.15

    # Protections
    @property
    def protections(self):
        return [
            {
            "method": "CooldownPeriod",
            "stop_duration_candles": 3
            }
        ]

    # TODO: Its not dry code!
    # Buy Hyperoptable Parameters/Spaces.
    buy_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-15", space='buy')
    buy_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='buy')
    buy_crossed_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-55", space='buy')

    buy_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='buy')
    buy_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-5", space='buy')
    buy_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="AROONOSC-5", space='buy')

    buy_operator0 = CategoricalParameter(operators, default="D", space='buy')
    buy_operator1 = CategoricalParameter(operators, default="<R", space='buy')
    buy_operator2 = CategoricalParameter(operators, default="=R", space='buy')

    buy_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS,  default=0.35, space='buy')
    buy_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.30, space='buy')
    buy_real_num2 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.00, space='buy')

    # Sell Hyperoptable Parameters/Spaces.
    sell_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-15", space='sell')
    sell_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='sell')
    sell_crossed_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-55", space='sell')

    sell_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='sell')
    sell_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-5", space='sell')
    sell_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="AROONOSC-5", space='sell')

    sell_operator0 = CategoricalParameter(operators, default="D", space='sell')
    sell_operator1 = CategoricalParameter(operators, default=">R", space='sell')
    sell_operator2 = CategoricalParameter(operators, default="=R", space='sell')

    sell_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.65, space='sell')
    sell_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.90, space='sell')
    sell_real_num2 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.00, space='sell')

    # Skippy learns from the best
    # DCA and Position Staking with help from Perkmeister on Discord

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Check if the entry already exists
        pair = metadata['pair']

        if not pair in self.custom_info:
            # Create empty entry for this pair {DATESTAMP}
            self.custom_info[pair] = [''] 

        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, current_profit: float,
                           **kwargs) -> bool:
        
        # Minimum 1% profit before Sell trigger
        if current_profit < 0.01:
            return False

        return True
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        
        # Split total stake amounst the bots and the max trades per bot
        custom_stake = self.wallets.get_total_stake_amount() / self.config['max_open_trades'] / (self.max_entry_position_adjustment + 1)
        if custom_stake >= min_stake:
            return custom_stake
        elif custom_stake < min_stake:
            return min_stake
        else:
            return proposed_stake

    # Skippy's has finally figured out how to stake positions.
    # You would think this is less difficult than forming a wormhole. You would be wrong..
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if(len(dataframe) < 1):
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        if(self.custom_info[trade.pair][self.DATESTAMP] != last_candle['date']):
            # Trigger once per cnadle
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle['date']

            # If current total profit is greater than value don't adjust.
            if current_profit > -0.01:
                return None

            # If last candle had 'buy' indicator adjust stake by original stake_amount
            if last_candle['buy'] > 0:
                filled_buys = trade.select_filled_orders('buy')
                count_of_buys = trade.nr_of_successful_buys
                try:
                    stake_amount = ((count_of_buys * self.dca_multiplier) + 1) * filled_buys[0].cost 
                    if stake_amount < min_stake: 
                        return min_stake
                    else:
                        return stake_amount
                except Exception as exception:
                    return None

        return None
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = list()

        # Buy Condition 0
        buy_indicator = self.buy_indicator0.value
        buy_crossed_indicator = self.buy_crossed_indicator0.value
        buy_operator = self.buy_operator0.value
        buy_real_num = self.buy_real_num0.value
        condition, dataframe = condition_generator(
            dataframe,
            buy_operator,
            buy_indicator,
            buy_crossed_indicator,
            buy_real_num
        )
        conditions.append(condition)

        # Buy Condition 1
        buy_indicator = self.buy_indicator1.value
        buy_crossed_indicator = self.buy_crossed_indicator1.value
        buy_operator = self.buy_operator1.value
        buy_real_num = self.buy_real_num1.value

        condition, dataframe = condition_generator(
            dataframe,
            buy_operator,
            buy_indicator,
            buy_crossed_indicator,
            buy_real_num
        )
        conditions.append(condition)

        # Buy Condition 2
        buy_indicator = self.buy_indicator2.value
        buy_crossed_indicator = self.buy_crossed_indicator2.value
        buy_operator = self.buy_operator2.value
        buy_real_num = self.buy_real_num2.value
        condition, dataframe = condition_generator(
            dataframe,
            buy_operator,
            buy_indicator,
            buy_crossed_indicator,
            buy_real_num
        )
        conditions.append(condition)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy']=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = list()
        
        # Sell Condition 0
        sell_indicator = self.sell_indicator0.value
        sell_crossed_indicator = self.sell_crossed_indicator0.value
        sell_operator = self.sell_operator0.value
        sell_real_num = self.sell_real_num0.value
        condition, dataframe = condition_generator(
            dataframe,
            sell_operator,
            sell_indicator,
            sell_crossed_indicator,
            sell_real_num
        )
        conditions.append(condition)

        # Sell Condition 1
        sell_indicator = self.sell_indicator1.value
        sell_crossed_indicator = self.sell_crossed_indicator1.value
        sell_operator = self.sell_operator1.value
        sell_real_num = self.sell_real_num1.value
        condition, dataframe = condition_generator(
            dataframe,
            sell_operator,
            sell_indicator,
            sell_crossed_indicator,
            sell_real_num
        )
        conditions.append(condition)

        # Sell Condition 2
        sell_indicator = self.sell_indicator2.value
        sell_crossed_indicator = self.sell_crossed_indicator2.value
        sell_operator = self.sell_operator2.value
        sell_real_num = self.sell_real_num2.value
        condition, dataframe = condition_generator(
            dataframe,
            sell_operator,
            sell_indicator,
            sell_crossed_indicator,
            sell_real_num
        )
        conditions.append(condition)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell']=1

        return dataframe