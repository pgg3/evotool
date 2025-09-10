class BaseConfig:
    def __init__(
            self, output_path,
            verbose: bool=True
    ):
        self.output_path = output_path
        self.verbose = verbose