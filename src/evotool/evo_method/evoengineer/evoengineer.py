from .run_config import EvoEngineerConfig
from ..base_method import Method

class EvoEngineer(Method):
    def __init__(self, config: EvoEngineerConfig):
        super().__init__(config)
        self.config = config