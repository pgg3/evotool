from evotool.task.python_task import Es1p1PythonAdapter
from .func_approxi_adapter import FuncApproxBaseAdapter


class Es1p1FuncApproxAdapter(FuncApproxBaseAdapter, Es1p1PythonAdapter):
    """ES(1+1) Adapter for function approximation tasks."""
    def __init__(self, task_info: dict):
        Es1p1PythonAdapter.__init__(self, task_info)