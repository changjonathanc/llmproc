"""Constants for environment information."""

import datetime
import getpass
import os
import platform

STANDARD_VAR_FUNCTIONS = {
    "working_directory": os.getcwd,
    "platform": lambda: platform.system().lower(),
    "date": lambda: datetime.datetime.now().strftime("%Y-%m-%d"),
    "python_version": platform.python_version,
    "hostname": platform.node,
    "username": getpass.getuser,
}

STANDARD_VAR_NAMES = list(STANDARD_VAR_FUNCTIONS)

RESERVED_KEYS = {"variables"}
