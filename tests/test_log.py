import random
import sys

import spade
from aioxmpp import JID

from macofl.log import (
    AlgorithmLogManager,
    GeneralLogManager,
    MessageLogManager,
    NnInferenceLogManager,
    NnTrainLogManager,
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
    for i in range(15):
        logger.log(
            current_round=100 + i, sender=sender, to=to, msg_type="SEND", size=250_000
        )
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")

    logger = NnTrainLogManager(extra_logger_name="test")
    for i in range(20):
        logger.log(
            current_round=100,
            agent=sender,
            seconds=random.random() * 10,
            epoch=i,
            accuracy=random.random(),
            loss=random.random(),
        )
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")

    logger = NnInferenceLogManager(extra_logger_name="test")
    for i in range(15):
        logger.log(
            current_round=100 + i,
            agent=sender,
            seconds=random.random() * 100,
            epochs=15,
            mean_training_accuracy=random.random(),
            mean_training_loss=random.random(),
            validation_accuracy=random.random(),
            validation_loss=random.random(),
            test_accuracy=random.random(),
            test_loss=random.random(),
        )
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")

    logger = AlgorithmLogManager(extra_logger_name="algorithm")
    for i in range(15):
        logger.log(current_round=100 + i, agent=sender, seconds=random.random() * 100)
    general_logger.info(f"Handlers: {logger.logger.handlers}")
    general_logger.info(f"Effective Level: {logger.logger.getEffectiveLevel()}")
