import asyncio
import logging
import sys
import traceback
import uuid

import spade
from aioxmpp import JID

from macofl.agent import CoordinatorAgent, LauncherAgent, ObserverAgent

# from macofl.agent.premiofl import AcolAgent
from macofl.log import GeneralLogManager, setup_loggers


async def main() -> None:
    uuid4_enabled = True
    xmpp_domain = "localhost"
    max_message_size = 250_000  # do not be close to 262 144
    number_of_agents = 8
    number_of_observers = 1

    uuid4 = str(uuid.uuid4()) if uuid4_enabled else ""

    logger = GeneralLogManager(extra_logger_name="main")

    logger.info("Starting...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"SPADE version: {spade.__version__}")
    logger.info(f"UUID4: {uuid4}")

    initial_agents: list[JID] = []
    for i in range(number_of_agents):
        initial_agents.append(JID.fromstr(f"a{i}__{uuid4}@{xmpp_domain}"))

    observer_jids: list[JID] = []
    for i in range(number_of_observers):
        observer_jids.append(JID.fromstr(f"o{i}__{uuid4}@{xmpp_domain}"))
    observers: list[ObserverAgent] = []

    logger.info("Initializating coordinator...")
    coordinator = CoordinatorAgent(
        jid=f"coordinator__{uuid4}@{xmpp_domain}",
        password="123",
        max_message_size=max_message_size,
        coordinated_agents=initial_agents,
        verify_security=False,
    )
    await asyncio.sleep(0.2)

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
        jid=f"launcher__{uuid4}@{xmpp_domain}",
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
        await asyncio.sleep(0.2)
        logger.info("Observers initialized.")

        logger.info("Starting coordinator...")
        await coordinator.start()
        await asyncio.sleep(0.2)
        logger.info("Coordinator initialized.")

        logger.info("Initializing launcher...")
        await launcher.start()
        await asyncio.sleep(0.2)
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
        logger.exception(e)
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
    try:
        setup_loggers(general_level=logging.INFO)
        spade.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
