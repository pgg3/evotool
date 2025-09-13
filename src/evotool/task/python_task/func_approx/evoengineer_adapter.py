from evotool.task.python_task import EvoEngineerPythonAdapter
from .func_approxi_adapter import FuncApproxBaseAdapter


class EvoEngineerFuncApproxAdapter(FuncApproxBaseAdapter, EvoEngineerPythonAdapter):
    """EvoEngineer Adapter for function approximation tasks."""
    def __init__(self, task_info: dict):
        EvoEngineerPythonAdapter.__init__(self, task_info)