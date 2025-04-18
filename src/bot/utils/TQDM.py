from tqdm import tqdm

from bot.config import TQDM_BAR_FORMAT

class CustomTQDM(tqdm):
    """modified from ultralytics"""
    def __init__(self, *args, **kwargs):
        """
        Initialize custom tqdm class with different default arguments.

        Note these can still be overridden when calling TQDM.
        """
        kwargs["disable"] = kwargs.get("disable", False)  # if passed
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)