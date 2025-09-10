class BaseConfig:
    def __init__(
            self,
            task_info:dict,
            output_path,
            verbose: bool=True
    ):
        self.task_info = task_info
        self.output_path = output_path
        self.verbose = verbose