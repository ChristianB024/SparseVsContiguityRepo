import logging


def configure_log_critical(log_file: str) -> None:
    """
    Creates a CRITICAL log file. The input should contain the name of the file without ".log" at the end.
    @param log_file: string which represents the name of the file without ".log" at the end
    """
    logging.basicConfig(
        filename="../logs/" + log_file + ".log",  # Specify the log file name
        level=logging.CRITICAL,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',  # Define the log format
        datefmt='%Y-%m-%d %H:%M:%S'  # Define the date-time format
    )


def configure_log_info(log_file: str) -> None:
    """
    Creates a INFO log file. The input should contain the name of the file without ".log" at the end.
    @param log_file:  string which represents the name of the file without ".log" at the end
    """
    logging.basicConfig(
        filename="../logs/" + log_file + ".log",  # Specify the log file name
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',  # Define the log format
        datefmt='%Y-%m-%d %H:%M:%S'  # Define the date-time format
    )


def configure_log_debug(log_file: str) -> None:
    """
    Creates a DEBUG log file. The input should contain the name of the file without ".log" at the end.
    @param log_file: string which represents the name of the file without ".log" at the end
    """
    logging.basicConfig(
        filename="../logs/" + log_file + ".log",  # Specify the log file name
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',  # Define the log format
        datefmt='%Y-%m-%d %H:%M:%S'  # Define the date-time format
    )
