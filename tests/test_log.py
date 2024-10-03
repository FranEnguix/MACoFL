import logging
import sys

import spade

from macofl.log import setup_loggers


def test_fill_logs():
    setup_loggers()

    general_logger = logging.getLogger("rf.log.test")
    general_logger.info("Starting test log...")
    general_logger.info(f"Python version: {sys.version}")
    general_logger.info(f"SPADE version: {spade.__version__}")
    general_logger.debug("Hello from the test sideee (debug edition)")
    general_logger.info(f"Handlers: {general_logger.handlers}")
    general_logger.info(f"Effective Level: {general_logger.getEffectiveLevel()}")

    logger = logging.getLogger("rf.message")
    logger.info("123,timestamp,sender,dest,type,size")
    logger.info("124,timestamp,sender,dest,type,size")
    logger.info("125,timestamp,sender,dest,type,size")
    logger.info("126,timestamp,sender,dest,type,size")
    general_logger.info(f"Handlers: {logger.handlers}")
    general_logger.info(f"Effective Level: {logger.getEffectiveLevel()}")

    logger = logging.getLogger("rf.nn")
    logger.info(
        "123,timestamp,agent,seconds,training_accuracy,training_loss,test_accuracy,test_loss"
    )
    logger.info(
        "124,timestamp,agent,seconds,training_accuracy,training_loss,test_accuracy,test_loss"
    )
    logger.info(
        "125,timestamp,agent,seconds,training_accuracy,training_loss,test_accuracy,test_loss"
    )
    logger.info(
        "126,timestamp,agent,seconds,training_accuracy,training_loss,test_accuracy,test_loss"
    )
    general_logger.info(f"Handlers: {logger.handlers}")
    general_logger.info(f"Effective Level: {logger.getEffectiveLevel()}")

    logger = logging.getLogger("rf.iteration")
    logger.info("123,timestamp,agent,seconds")
    logger.info("124,timestamp,agent,seconds")
    logger.info("125,timestamp,agent,seconds")
    logger.info("126,timestamp,agent,seconds")
    general_logger.info(f"Handlers: {logger.handlers}")
    general_logger.info(f"Effective Level: {logger.getEffectiveLevel()}")
