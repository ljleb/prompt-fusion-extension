import dataclasses
import torch
from modules import prompt_parser
from typing import NamedTuple, Union


class InterpolationParams(NamedTuple):
    t: float
    step: int
    total_steps: int
    slerp_scale: float
    slerp_epsilon: float


class InterpolationTensor:
    def __init__(self, sub_tensors, interpolation_function):
        self.__sub_tensors = sub_tensors
        self.__interpolation_function = interpolation_function

    def interpolate(self, params: InterpolationParams, origin_cond, empty_cond):
        cond_delta = self.interpolate_cond_delta_rec(params, origin_cond, empty_cond)
        return (cond_delta + origin_cond.extend_like(cond_delta, empty_cond)).to(dtype=origin_cond.dtype)

    def interpolate_cond_delta_rec(self, params: InterpolationParams, origin_cond, empty_cond):
        if self.__interpolation_function is None:
            return self.to_cond_delta(params.step, origin_cond, empty_cond)

        control_points = [
            sub_tensor.interpolate_cond_delta_rec(params, origin_cond, empty_cond)
            for sub_tensor in self.__sub_tensors
        ]

        CondWrapper, control_points_values = conds_to_cp_values(control_points)
        return CondWrapper.from_cp_values(self.__interpolation_function(control_points, params) for control_points in control_points_values)

    def to_cond_delta(self, step, origin_cond, empty_cond):
        schedule = None
        for schedule in self.__sub_tensors:
            if schedule.end_at_step >= step:
                break

        return schedule.cond.extend_like(origin_cond, empty_cond).to(dtype=torch.double) - origin_cond.extend_like(schedule.cond, empty_cond).to(dtype=torch.double)


def conds_to_cp_values(conds):
    CondWrapper = type(conds[0])
    cp_values = [
        cond.to_cp_values()
        for cond in conds
    ]
    return CondWrapper, [
        [v[i] for v in cp_values]
        for i in range(len(cp_values[0]))
    ]


class InterpolationTensorBuilder:
    def __init__(self, tensor=None, prompt_database=None, interpolation_functions=None):
        self.__indices_tensor = tensor if tensor is not None else 0
        self.__prompt_database = prompt_database if prompt_database is not None else ['']
        self.__interpolation_functions = interpolation_functions if interpolation_functions is not None else []

    def append(self, suffix):
        for i in range(len(self.__prompt_database)):
            self.__prompt_database[i] += suffix

    def extrude(self, tensor_updaters, interpolation_function):
        extruded_indices_tensor = []
        extruded_prompt_database = []
        extruded_interpolation_functions = []

        for update_tensor in tensor_updaters:
            nested_tensor_builder = InterpolationTensorBuilder(
                self.__indices_tensor,
                self.__prompt_database[:],
                interpolation_functions=[])

            update_tensor(nested_tensor_builder)

            extruded_indices_tensor.append(InterpolationTensorBuilder.__offset_tensor(
                tensor=nested_tensor_builder.__indices_tensor,
                offset=len(extruded_prompt_database)))
            extruded_prompt_database.extend(nested_tensor_builder.__prompt_database)
            extruded_interpolation_functions.append(nested_tensor_builder.__interpolation_functions)

        self.__indices_tensor = extruded_indices_tensor
        self.__prompt_database[:] = extruded_prompt_database
        self.__interpolation_functions.insert(0, (interpolation_function, extruded_interpolation_functions))

    def get_prompt_database(self):
        return self.__prompt_database

    @staticmethod
    def __offset_tensor(tensor, offset):
        try:
            return tensor + offset

        except TypeError:
            return [InterpolationTensorBuilder.__offset_tensor(e, offset) for e in tensor]

    def build(self, conds, empty_cond):
        max_cond_size = self.__max_cond_size(conds)
        conds = self.__resize_uniformly(conds, max_cond_size, empty_cond)
        return InterpolationTensorBuilder.__build_conditionings_tensor(self.__indices_tensor, self.__interpolation_functions, conds)

    @staticmethod
    def __build_conditionings_tensor(tensor, int_funcs, conds):
        if type(tensor) is int:
            return InterpolationTensor(conds[tensor], None)
        else:
            int_func, nested_int_funcs = int_funcs[0]
            return InterpolationTensor(
                [
                    InterpolationTensorBuilder.__build_conditionings_tensor(sub_tensor, nested_int_funcs + int_funcs[1:], conds)
                    for sub_tensor, nested_int_funcs in zip(tensor, nested_int_funcs)
                ],
                int_func,
            )

    def __resize_uniformly(self, conds, max_cond_size: int, empty_cond):
        return [
            [
                prompt_parser.ScheduledPromptConditioning(
                    cond=schedule.cond.resize_schedule(max_cond_size, empty_cond),
                    end_at_step=schedule.end_at_step
                )
                for schedule in schedules
            ]
            for schedules in conds
        ]

    @staticmethod
    def __max_cond_size(conds):
        return max(schedule.cond.size(0)
                   for schedules in conds
                   for schedule in schedules)


@dataclasses.dataclass
class DictCondWrapper:
    original_cond: dict

    @staticmethod
    def from_cp_values(cp_values):
        return DictCondWrapper({
            k: v
            for k, v in zip(('crossattn', 'vector'), cp_values)
        })

    def size(self, *args, **kwargs):
        return self.original_cond['crossattn'].size(*args, **kwargs)

    def extend_like(self, that, empty):
        missing_size = max(0, that.size(0) - self.size(0)) // 77
        extended = DictCondWrapper(self.original_cond.copy())
        extended.original_cond['crossattn'] = torch.concatenate([self.original_cond['crossattn']] + [empty.original_cond['crossattn']] * missing_size)
        return extended

    def resize_schedule(self, target_size, empty_cond):
        cond_missing_size = (target_size - self.size(0)) // 77
        if cond_missing_size <= 0:
            return self

        resized_cond = self.original_cond.copy()
        resized_cond['crossattn'] = torch.concatenate([self.original_cond['crossattn']] + [empty_cond.original_cond['crossattn']] * cond_missing_size)
        return DictCondWrapper(resized_cond)

    def to_cp_values(self):
        return list(self.original_cond.values())

    def to(self, dtype: Union[dict, torch.dtype]):
        if not isinstance(dtype, dict):
            dtype = {
                k: dtype
                for k in self.original_cond.keys()
            }
        return DictCondWrapper({
            k: v.to(dtype=dtype[k])
            for k, v in self.original_cond.items()
        })

    @property
    def dtype(self):
        return {
            k: v.dtype
            for k, v in self.original_cond.items()
        }

    def __sub__(self, that):
        return DictCondWrapper({
            k: v - that.original_cond[k]
            for k, v in self.original_cond.items()
        })

    def __add__(self, that):
        return DictCondWrapper({
            k: v + that.original_cond[k]
            for k, v in self.original_cond.items()
        })

    def __eq__(self, that):
        return all((self.original_cond[k] == that.original_cond[k]).all() for k in self.original_cond.keys())


@dataclasses.dataclass
class TensorCondWrapper:
    original_cond: torch.Tensor

    @staticmethod
    def from_cp_values(cp_values):
        return TensorCondWrapper(next(iter(cp_values)))

    def size(self, *args, **kwargs):
        return self.original_cond.size(*args, **kwargs)

    def extend_like(self, that, empty):
        missing_size = max(0, that.size(0) - self.original_cond.size(0)) // 77
        return TensorCondWrapper(torch.concatenate([self.original_cond] + [empty.original_cond] * missing_size))

    def resize_schedule(self, target_size, empty_cond):
        cond_missing_size = (target_size - self.original_cond.size(0)) // 77
        if cond_missing_size <= 0:
            return self

        return TensorCondWrapper(torch.concatenate([self.original_cond] + [empty_cond.original_cond] * cond_missing_size))

    def to_cp_values(self):
        return [self.original_cond]

    def to(self, dtype: torch.dtype):
        return TensorCondWrapper(self.original_cond.to(dtype=dtype))

    @property
    def dtype(self):
        return self.original_cond.dtype

    def __sub__(self, that):
        return TensorCondWrapper(self.original_cond - that.original_cond)

    def __add__(self, that):
        return TensorCondWrapper(self.original_cond + that.original_cond)

    def __eq__(self, that):
        return (self.original_cond == that.original_cond).all()
