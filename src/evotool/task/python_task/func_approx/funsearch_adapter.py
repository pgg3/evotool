from evotool.task.python_task import FunSearchPythonAdapter
from .func_approxi_adapter import FuncApproxBaseAdapter


class FunSearchFuncApproxAdapter(FuncApproxBaseAdapter, FunSearchPythonAdapter):
    """FunSearch Adapter for function approximation tasks."""
    def __init__(self, task_info: dict):
        FunSearchPythonAdapter.__init__(self, task_info)
