import time
from decimal import Decimal
from typing import List, Optional

import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.smart_components.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.smart_components.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.smart_components.executors.position_executor.data_types import TrailingStop
from hummingbot.smart_components.order_level_distributions.distributions import Distributions
from hummingbot.smart_components.utils.peak_analysis_utils import (
    calculate_prominence,
    find_price_peaks,
    hierarchical_clustering,
)


class MarroqControllerConfig(MarketMakingControllerConfigBase):
    controller_name = "marroq"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ", )
    )
    candles_trading_pair: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ", )
    )
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    cluster_window: int = Field(
        default=1500,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the cluster window size: ",
            prompt_on_new=True))
    cluster_prominence: float = Field(
        default=0.02,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the cluster prominence percentage: "))
    cluster_width: int = Field(
        default=5,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the cluster width: "))
    dca_levels: int = Field(
        default=4,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the number of DCA levels: ",
            prompt_on_new=True))
    dca_amounts_distribution: List[Decimal] = Field(
        default=Distributions.geometric(4, 0.01, 2),
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=False))
    executor_activation_bounds: Optional[List[Decimal]] = Field(
        default=None,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the activation bounds for the orders "
                              "(e.g., 0.01 activates the next order when the price is closer than 1%): ",
            prompt_on_new=False))

    @validator("executor_activation_bounds", pre=True, always=True)
    def parse_activation_bounds(cls, v):
        if isinstance(v, list):
            return [Decimal(val) for val in v]
        elif isinstance(v, str):
            if v == "":
                return None
            return [Decimal(val) for val in v.split(",")]
        return v

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class MarroqController(MarketMakingControllerBase):
    def __init__(self, config: MarroqControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.cluster_window
        self.dca_amounts_pct = [amount / sum(config.dca_amounts_distribution) for amount in config.dca_amounts_distribution]
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    def find_and_cluster_peaks(self, candles, prominence_percentage=0.02, distance=4, num_clusters=3):
        # Calculate prominence
        prominence = calculate_prominence(candles, prominence_percentage=prominence_percentage)
        # Find peaks
        high_peaks, low_peaks = find_price_peaks(candles, prominence, distance=distance)  # Adjust distance as needed
        # Cluster peaks
        high_peak_prices = candles['high'].iloc[high_peaks].values
        low_peak_prices = candles['low'].iloc[low_peaks].values
        high_clusters, _ = hierarchical_clustering(high_peak_prices, num_clusters)
        low_clusters, _ = hierarchical_clustering(low_peak_prices, num_clusters)
        return high_clusters, low_clusters

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        trade_type = self.get_trade_type_from_level_id(level_id)

        # Get candles and cluster levels
        candles = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        high_clusters, low_clusters = self.find_and_cluster_peaks(
            candles,
            prominence_percentage=self.config.cluster_prominence,
            distance=self.config.cluster_width,
            num_clusters=self.config.dca_levels + 1
        )

        # Filter clusters based on current price
        relevant_high_clusters = [c for c in high_clusters if c > price]
        relevant_low_clusters = [c for c in low_clusters if c < price]

        # Determine relevant clusters based on trade type
        selected_clusters = relevant_low_clusters if trade_type == TradeType.BUY else relevant_high_clusters

        # Calculate amounts for each level
        if len(selected_clusters) + 1 < len(self.dca_amounts_pct):
            adjusted_amounts_pct = self.dca_amounts_pct[:len(selected_clusters) + 1]
        else:
            adjusted_amounts_pct = self.dca_amounts_pct

        # Calculate total range
        total_range = Decimal(candles['high'].max() - candles['low'].min())

        prices = [price] + [Decimal(cluster_level) for cluster_level in selected_clusters]
        amounts_quote = [amount * pct * price for pct, price in zip(adjusted_amounts_pct, prices)]

        breakeven_price = sum(p * a for p, a in zip(prices, amounts_quote)) / sum(amounts_quote)

        # Determine stop-loss and take-profit
        if trade_type == TradeType.BUY:
            sl_price = Decimal(min(selected_clusters)) - (total_range * self.config.stop_loss)
            sl_pct = (breakeven_price - sl_price) / breakeven_price
            trailing_stop_price = Decimal(min(relevant_high_clusters))
        else:
            sl_price = Decimal(max(selected_clusters)) + (total_range * self.config.stop_loss)
            sl_pct = (sl_price - breakeven_price) / breakeven_price
            trailing_stop_price = Decimal(max(relevant_low_clusters))
        return DCAExecutorConfig(
            timestamp=time.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            prices=prices,
            amounts_quote=amounts_quote,
            leverage=self.config.leverage,
            side=trade_type,
            time_limit=self.config.time_limit,
            stop_loss=sl_pct,
            trailing_stop=TrailingStop(activation_price=(price - trailing_stop_price) / price,
                                       trailing_delta=self.config.trailing_stop.trailing_delta),
            activation_bounds=self.config.executor_activation_bounds,
            take_profit=self.config.take_profit,
        )
