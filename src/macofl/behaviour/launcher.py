import asyncio

from typing import Optional, TYPE_CHECKING

from spade.behaviour import CyclicBehaviour, OneShotBehaviour

if TYPE_CHECKING:
    from macofl.agent import LauncherAgent


class LaunchAgentsBehaviour(OneShotBehaviour):

    async def run(self) -> None:
        agent: LauncherAgent = self.agent
        await agent.launch_agents()
        agent.logger.info("Agents launched.")


class Wait(CyclicBehaviour):

    def __init__(self, delay: Optional[float] = 5):
        self.delay = delay
        super().__init__()

    async def run(self) -> None:
        await asyncio.sleep(self.delay)
