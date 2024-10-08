import random
import sys

import spade
from aioxmpp import JID

from macofl.log import (
    AlgorithmLogManager,
    GeneralLogManager,
    MessageLogManager,
    NnLogManager,
    setup_loggers,
)


def test_fill_logs():
    setup_loggers()

    general_logger = GeneralLogManager(extra_logger_name="test")
    general_logger.info("Starting test log...")
    general_logger.info(f"Python version: {sys.version}")
    general_logger.info(f"SPADE version: {spade.__version__}")
    general_logger.debug("Hello from the test sideee (debug edition)")
    general_logger.info(f"Handlers: {general_logger.logger.handlers}")
    general_logger.info(f"Effective Level: {general_logger.logger.getEffectiveLevel()}")

    sender = JID.fromstr("sender@localhost")
    to = JID.fromstr("to@localhost")
    logger = MessageLogManager(extra_logger_name="test")
    logger.log(iteration_id=123, sender=sender, to=to, msg_type="SEND", size=250_000)
    logger.log(iteration_id=124, sender=sender, to=to, msg_type="SEND", size=250_000)
    logger.log(iteration_id=125, sender=sender, to=to, msg_type="SEND", size=250_000)
    logger.log(iteration_id=126, sender=sender, to=to, msg_type="SEND", size=250_000)
    logger.log(iteration_id=127, sender=sender, to=to, msg_type="SEND", size=250_000)
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")

    logger = NnLogManager(extra_logger_name="test")
    logger.log(
        iteration_id=123,
        agent=sender,
        seconds=400.3567,
        training_accuracy=random.random(),
        training_loss=random.random(),
        test_accuracy=random.random(),
        test_loss=random.random(),
    )
    logger.log(
        iteration_id=124,
        agent=sender,
        seconds=345.345,
        training_accuracy=random.random(),
        training_loss=random.random(),
        test_accuracy=random.random(),
        test_loss=random.random(),
    )
    logger.log(
        iteration_id=125,
        agent=sender,
        seconds=435.3452,
        training_accuracy=random.random(),
        training_loss=random.random(),
        test_accuracy=random.random(),
        test_loss=random.random(),
    )
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")

    logger = AlgorithmLogManager(extra_logger_name="algorithm")
    logger.log(iteration_id=123, agent=sender, seconds=234.3456)
    logger.log(iteration_id=124, agent=sender, seconds=244.45)
    logger.log(iteration_id=125, agent=sender, seconds=546.2345)
    logger.log(iteration_id=126, agent=sender, seconds=245.435)
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")
