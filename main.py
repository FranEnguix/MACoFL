import asyncio
import spade
import logging
import traceback
import sys

from aioxmpp import JID

from macofl.log import setup_loggers
from macofl.agent import CoordinatorAgent, LauncherAgent, ObserverAgent


async def main():
    xmpp_domain = "localhost"
    agent_name = "ag"
    max_message_size = 250_000  # do not be close to 262 144
    number_of_agents = 10
    number_of_observers = 1

    logger = logging.getLogger("rf.log.main")

    logger.info("Starting...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"SPADE version: {spade.__version__}")

    initial_agents: list[JID] = []
    for i in range(number_of_agents):
        initial_agents.append(JID.fromstr(f"{agent_name}_{i}@{xmpp_domain}"))

    observer_jids: list[JID] = []
    for i in range(number_of_observers):
        observer_jids.append(JID.fromstr(f"obs_{i}@{xmpp_domain}"))
    observers: list[ObserverAgent] = []

    logger.info("Initializating coordinator...")
    coordinator = CoordinatorAgent(
        jid=f"coordinator@{xmpp_domain}",
        password="123",
        max_message_size=max_message_size,
        coordinated_agents=initial_agents,
        verify_security=False,
    )
    await asyncio.sleep(5)

    for obs_jid in observer_jids:
        obs = ObserverAgent(
            jid=str(obs_jid),
            password="123",
            max_message_size=max_message_size,
            verify_security=False,
        )
        observers.append(obs)

    logger.info("Initializating launcher...")
    launcher = LauncherAgent(
        jid=f"launcher@{xmpp_domain}",
        password="123",
        max_message_size=max_message_size,
        agents_coordinator=coordinator.jid,
        agents_observers=observer_jids,
        agents_to_launch=initial_agents,
        verify_security=False,
    )

    try:
        logger.info("Starting observers...")
        for observer in observers:
            await observer.start()
        await asyncio.sleep(5)
        logger.info("Observers initialized.")

        logger.info("Starting coordinator...")
        await coordinator.start()
        await asyncio.sleep(5)
        logger.info("Coordinator initialized.")

        logger.info("Initializing launcher...")
        await launcher.start()
        await asyncio.sleep(5)
        logger.info("Launcher initialized.")

        logger.info("Waiting for coordinator...")
        await spade.wait_until_finished(coordinator)
        logger.info("Coordinator finished.")

        logger.info("Waiting for launcher...")
        await spade.wait_until_finished(launcher)
        logger.info("Launcher finished.")

        logger.info("Waiting for agents...")
        await spade.wait_until_finished(launcher.agents)
        logger.info("Agents finished.")

    except Exception as e:
        logger.error(e)
        traceback.print_exc()

    finally:
        logger.info("Stopping...")
        if coordinator.is_alive():
            await coordinator.stop()
        if launcher.is_alive():
            await launcher.stop()
        for ag in launcher.agents:
            if ag.is_alive():
                await ag.stop()
        logger.info("Run finished.")


if __name__ == "__main__":
    setup_loggers(general_level=logging.INFO)
    spade.run(main())
