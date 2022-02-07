import logging
import logging.config
import yaml

# Level       |   Numeric value
# ⸻⸻⸻⸻⸻⸻⸻
# CRITICAL    |   50
# ERROR       |   40
# WARNING     |   30
# INFO        |   20
# DEBUG       |   10
# NOTSET      |   0


class Logger:
    def __init__(self):
        with open("logConfig.yml","r") as stream:
            logging.config.dictConfig(yaml.load(stream,Loader=yaml.FullLoader))
        self.ERROR = logging.getLogger("error").error
        self.INFO = logging.getLogger("info").info
