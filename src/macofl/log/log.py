import logging


def setup_loggers() -> None:
    # Set up the base format and log levels for the different loggers
    formatter = logging.Formatter(
        "%(asctime)s; %(name)s; %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # General agent log
    rf_console_log_handler = logging.StreamHandler()
    rf_console_log_handler.setLevel(logging.DEBUG)
    rf_console_log_handler.setFormatter(formatter)

    rf_log_handler = logging.FileHandler("general.log")
    rf_log_handler.setLevel(logging.DEBUG)
    rf_log_handler.setFormatter(formatter)

    # Message log
    message_log_handler = logging.FileHandler("messages.log")
    message_log_handler.setLevel(logging.INFO)
    message_log_handler.setFormatter(formatter)

    # Neural network accuracy log
    accuracy_log_handler = logging.FileHandler("accuracy.log")
    accuracy_log_handler.setLevel(logging.INFO)
    accuracy_log_handler.setFormatter(formatter)

    # Attach handlers to the loggers for each agent category
    base_logger = logging.getLogger("rf.log")
    base_logger.setLevel(logging.INFO)
    base_logger.addHandler(rf_log_handler)
    base_logger.addHandler(rf_console_log_handler)

    message_logger = logging.getLogger("rf.message")
    message_logger.setLevel(logging.INFO)
    message_logger.addHandler(message_log_handler)

    accuracy_logger = logging.getLogger("rf.accuracy")
    accuracy_logger.setLevel(logging.INFO)
    accuracy_logger.addHandler(accuracy_log_handler)
