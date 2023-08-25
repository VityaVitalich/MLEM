import logging
from argparse import ArgumentParser


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=("debug", "info", "warning", "error", "critical"),
        help="Python native logging level",
    )
    args = parser.parse_args()


    logger.setLevel(getattr(logging, args.log_level.upper()))
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:\t%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

