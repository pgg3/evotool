
from evotool.task.python_task import EohPythonAdapter
from .func_approxi_adapter import FuncApproxBaseAdapter


class EohFuncApproxAdapter(FuncApproxBaseAdapter, EohPythonAdapter):
    """EOH Adapter for Function Approximation Task"""
    def __init__(self, task_info: dict):
        EohPythonAdapter.__init__(self, task_info)