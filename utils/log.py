import sys
import logging

LOGFILE = "/tmp/log.txt"

WRITE_TO_FILE = True

format = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'

logger = logging.getLogger('')
logger.setLevel(logging.INFO)


# create console handler with a higher log level
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(ch)

if WRITE_TO_FILE:
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOGFILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
