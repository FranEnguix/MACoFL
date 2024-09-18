import asyncio
import spade
import logging
import traceback
import sys

from aioxmpp import JID

from macofl.log import setup_loggers
from macofl.agent import CoordinatorAgent, LauncherAgent


async def main():
    xmpp_domain = "localhost"
    agent_name = "akkk"
    max_message_size = 250_000  # do not be close to 262 144
    number_of_agents = 10

    logger = logging.getLogger("rf.log.main")

    initial_agents: list[JID] = []
    for i in range(number_of_agents):
        initial_agents.append(JID.fromstr(f"{agent_name}_{i}@{xmpp_domain}"))

    logger.info("Initializating coordinator...")
    coordinator = CoordinatorAgent(
        jid=f"coordinator@{xmpp_domain}",
        password="123",
        max_message_size=max_message_size,
        coordinated_agents=initial_agents,
        verify_security=False,
    )

    logger.info("Initializating launcher...")
    launcher = LauncherAgent(
        jid=f"launcher@{xmpp_domain}",
        password="123",
        max_message_size=max_message_size,
        agents_coordinator=coordinator.jid,
        agents_observers=[],
        agents_to_launch=initial_agents,
        verify_security=False,
    )

    try:
        logger.info("Starting coordinator...")
        await coordinator.start()
        await asyncio.sleep(5)
        logger.info("Coordinator initialized.")

        logger.info("Initializing launcher...")
        await launcher.start()
        await asyncio.sleep(5)
        logger.info("Launcher initialized.")

        logger.info("Waiting for agents...")
        await spade.wait_until_finished([launcher])
        logger.info("Launcher finished.")
        await spade.wait_until_finished(launcher.agents)
        logger.info("Agents finished.")

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        print("Force Stopping...")

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
    print("Starting...")
    print(f"Python version: {sys.version}")
    print(f"SPADE version: {spade.__version__}")
    setup_loggers()
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     filename="agents.log",
    #     filemode="a",  # Append mode
    # )
    # for handler in logging.root.handlers:
    #     handler.addFilter(logging.Filter(""))
    # logger = logging.getLogger(f"rf.sys.{__name__}")
    # logger.info(f"SPADE version: {spade.__version__}")
    spade.run(main())
