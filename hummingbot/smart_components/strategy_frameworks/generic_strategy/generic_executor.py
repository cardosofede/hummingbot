import time
from typing import List, Optional

from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.smart_components.executors.arbitrage_executor.arbitrage_executor import ArbitrageExecutor
from hummingbot.smart_components.executors.arbitrage_executor.data_types import ArbitrageExecutorConfig
from hummingbot.smart_components.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.smart_components.executors.dca_executor.dca_executor import DCAExecutor
from hummingbot.smart_components.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.smart_components.executors.position_executor.position_executor import PositionExecutor
from hummingbot.smart_components.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
    StoreExecutorAction,
)
from hummingbot.smart_components.models.executors_info import ExecutorHandlerInfo, ExecutorInfo
from hummingbot.smart_components.strategy_frameworks.executor_handler_base import ExecutorHandlerBase
from hummingbot.smart_components.strategy_frameworks.generic_strategy.generic_controller import GenericController
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class GenericExecutor(ExecutorHandlerBase):
    """
    Generic executor handler for a strategy.
    """

    def on_stop(self):
        """Actions to perform on stop."""
        pass

    def stop(self):
        self.controller.stop()
        self.stop_all_executors()

    def stop_all_executors(self):
        for executor in self.position_executors + self.dca_executors + self.arbitrage_executors:
            executor.early_stop()

    def __init__(self, strategy: ScriptStrategyBase, controller: GenericController, update_interval: float = 1.0,
                 executors_update_interval: float = 1.0):
        super().__init__(strategy, controller, update_interval, executors_update_interval)
        self.controller = controller
        self.position_executors = []
        self.dca_executors = []
        self.arbitrage_executors = []

    async def control_task(self):
        """
        Override the control task to implement the dynamic behavior.
        """
        # Collect data for active executors
        executor_handler_report: ExecutorHandlerInfo = self.get_executor_handler_report()

        # Determine actions based on the collected data
        await self.controller.update_executor_handler_report(executor_handler_report)
        actions: Optional[List[ExecutorAction]] = await self.controller.determine_actions()

        # Execute actions
        await self.execute_actions(actions)

    def get_active_position_executors(self) -> List[ExecutorInfo]:
        """
        Get the active position executors.
        """
        return [executor.executor_info for executor in self.position_executors if not executor.is_closed]

    def get_closed_position_executors(self) -> List[ExecutorInfo]:
        """
        Get the closed position executors.
        """
        return [executor.executor_info for executor in self.position_executors if executor.is_closed]

    def get_active_dca_executors(self) -> List[ExecutorInfo]:
        """
        Get the active DCA executors.
        """
        return [executor.executor_info for executor in self.dca_executors if not executor.is_closed]

    def get_closed_dca_executors(self) -> List[ExecutorInfo]:
        """
        Get the closed DCA executors.
        """
        return [executor.executor_info for executor in self.dca_executors if executor.is_closed]

    def get_active_arbitrage_executors(self) -> List[ExecutorInfo]:
        """
        Get the active arbitrage executors.
        """
        return [executor.executor_info for executor in self.arbitrage_executors if not executor.is_closed]

    def get_closed_arbitrage_executors(self) -> List[ExecutorInfo]:
        """
        Get the closed arbitrage executors.
        """
        return [executor.executor_info for executor in self.arbitrage_executors if executor.is_closed]

    def get_executor_handler_report(self):
        """
        Compute information about executors.
        """
        executor_handler_report = ExecutorHandlerInfo(
            controller_id=self.controller.config.id,  # TODO: placeholder to refactor using async.Queue to communicate with the controller
            timestamp=time.time(),
            status=self.status,
            active_position_executors=self.get_active_position_executors(),
            closed_position_executors=self.get_closed_position_executors(),
            active_dca_executors=self.get_active_dca_executors(),
            closed_dca_executors=self.get_closed_dca_executors(),
            active_arbitrage_executors=self.get_active_arbitrage_executors(),
            closed_arbitrage_executors=self.get_closed_arbitrage_executors(),
        )
        return executor_handler_report

    async def execute_actions(self, actions: List[ExecutorAction]):
        """
        Execute the actions and return the status for each action.
        """
        for action in actions:
            if isinstance(action, CreateExecutorAction):
                # TODO:
                #  - evaluate if the id is unique
                #  - refactor to store the executor by controller id
                self.create_executor(action.executor_config)
            elif isinstance(action, StopExecutorAction):
                executor = self.get_executor_by_id(action.executor_id)
                if executor and executor.is_active:
                    executor.early_stop()
            elif isinstance(action, StoreExecutorAction):
                executor = self.get_executor_by_id(action.executor_id)
                if executor and executor.is_closed:
                    MarketsRecorder.get_instance().store_executor(executor)
                    self.dca_executors.remove(executor)
            else:
                raise ValueError(f"Unknown action type {type(action)}")

    def create_executor(self, executor_config):
        """
        Create an executor.
        """
        # TODO: refactor to use a factory
        if isinstance(executor_config, PositionExecutorConfig):
            executor = PositionExecutor(self.strategy, executor_config, self.executors_update_interval)
            executor.start()
            self.position_executors.append(executor)
            self.logger().debug(f"Created position executor {executor_config.id}")
        elif isinstance(executor_config, DCAExecutorConfig):
            executor = DCAExecutor(self.strategy, executor_config, self.executors_update_interval)
            executor.start()
            self.dca_executors.append(executor)
            self.logger().debug(f"Created DCA executor {executor_config.id}")
        elif isinstance(executor_config, ArbitrageExecutorConfig):
            executor = ArbitrageExecutor(self.strategy, executor_config, self.executors_update_interval)
            executor.start()
            self.arbitrage_executors.append(executor)
            self.logger().debug(f"Created arbitrage executor {executor_config.id}")

    def get_executor_by_id(self, executor_id: str):
        """
        Get the executor by id.
        """
        all_executors = self.position_executors + self.dca_executors + self.arbitrage_executors
        executor = next((executor for executor in all_executors if executor.config.id == executor_id), None)
        return executor

    def on_start(self):
        if self.controller.is_perpetual:
            self.set_leverage_and_position_mode()

    def set_leverage_and_position_mode(self):
        connector = self.strategy.connectors[self.controller.config.exchange]
        connector.set_position_mode(self.controller.config.position_mode)
        connector.set_leverage(trading_pair=self.controller.config.trading_pair, leverage=self.controller.config.leverage)

    def to_format_status(self) -> str:
        """
        Base status for executor handler.
        """
        lines = []
        lines.extend(self.controller.to_format_status())
        return "\n".join(lines)