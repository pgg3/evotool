from typing import Literal

from evotool.evo_method.base_run_state_dict import BaseRunStateDict

class AiCudaEngineerRunStateDict(BaseRunStateDict):
    def __init__(
            self,
            task_info:dict,
            run_stage: Literal["0", "1", "2"] = "0",
            evo_gen_i: int = 0,
            optimization_history: list = None,
            usage_history: dict = None,
            is_done: bool=False
    ):
        super().__init__(task_info)

        self.run_stage = run_stage  # 0: conversion, 1:translation

        self.evo_gen_i = evo_gen_i
        self.optimization_history = optimization_history or []

        self.usage_history = usage_history or {}

        self.is_done = is_done

    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary"""
        return {
            'task_info': self.task_info,
            'usage_history': self.usage_history,
            'run_stage': self.run_stage,
            'evo_gen_i': self.evo_gen_i,
            'optimization_history': self.optimization_history,
            'is_done': self.is_done
        }

    @classmethod
    def from_json(cls, data: dict) -> 'AiCudaEngineerRunStateDict':
        """Create instance from JSON data"""
        instance = cls(
            task_info=data['task_info'],
            run_stage=data.get('run_stage', "0"),  # type: ignore
            evo_gen_i=data.get('evo_gen_i', 0),
            optimization_history=data.get('optimization_history', []),
            usage_history=data.get('usage_history', {}),
            is_done=data.get('is_done', False)
        )
        instance.usage_history = data.get('usage_history', {})
        return instance
