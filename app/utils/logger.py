import logging


class Logger:
    """
    Logger singleton class. Allows for instantiation in any file. Any logging exports to
    'experiment.log' file.

    Attributes:
        _instance: reference to logger instance
    """
    _instance = None

    @classmethod
    def get_logger(cls, name="sim_logger", log_file="experiment.log"):
        if cls._instance is None:
            cls._instance = logging.getLogger(name)
            cls._instance.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('{asctime} {name} {levelname:6s} {message}', style='{')
            fh.setFormatter(formatter)
            cls._instance.addHandler(fh)
        return cls._instance
