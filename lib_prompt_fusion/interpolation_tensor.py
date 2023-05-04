import torch
from modules import prompt_parser, shared
from typing import NamedTuple


class InterpolationParams(NamedTuple):
    t: float
    step: int
    slerp_scale: float
    slerp_epsilon: float


class InterpolationTensor:
    def __init__(self, conditionings_tensor, interpolation_functions, empty_cond: torch.Tensor):
        self.__conditionings_tensor = conditionings_tensor
        self.__interpolation_functions = interpolation_functions
        self.__empty_cond = empty_cond

    def interpolate(self, params: InterpolationParams, origin_cond: torch.Tensor):
        cond = self.interpolate_rec(params, 0, origin_cond)
        return self.__resize_cond_like(origin_cond, cond) + cond

    def interpolate_rec(self, params: InterpolationParams, axis: int, origin_cond: torch.Tensor):
        tensor_axes = len(self.__interpolation_functions) - axis
        if tensor_axes == 0:
            if type(self.__conditionings_tensor) is not list:
                return self.__conditionings_tensor

            schedule = None
            for schedule in self.__conditionings_tensor:
                if schedule.end_at_step >= params.step:
                    break

            assert schedule is not None, "hmm! that's a weird one. devs expected this to work for some reason KEKW"
            return self.__resize_cond_like(schedule.cond, origin_cond) - self.__resize_cond_like(origin_cond, schedule.cond)

        interpolation_function, control_points_functions = self.__interpolation_functions[axis]
        if tensor_axes == 1:
            control_points = list(self.__conditionings_tensor)
        else:
            control_points = [InterpolationTensor(sub_tensor, self.__interpolation_functions, self.__empty_cond).interpolate_rec(params, axis + 1, origin_cond)
                              for sub_tensor in self.__conditionings_tensor]

        for i, nested_functions in enumerate(control_points_functions):
            control_points[i] = InterpolationTensor(control_points[i], nested_functions, self.__empty_cond).interpolate_rec(params, 0, origin_cond)

        return interpolation_function(control_points, params)

    def __resize_cond_like(self, cond_to_resize, reference_cond):
        missing_size = max(0, reference_cond.size(0) - cond_to_resize.size(0)) // 77
        return torch.concatenate([cond_to_resize] + [self.__empty_cond] * missing_size)


class InterpolationTensorBuilder:
    def __init__(self, tensor=None, prompt_database=None, interpolation_functions=None, empty_cond=None):
        self.__indices_tensor = tensor if tensor is not None else 0
        self.__prompt_database = prompt_database if prompt_database is not None else ['']
        self.__interpolation_functions = interpolation_functions if interpolation_functions is not None else []
        self.__empty_cond = empty_cond if empty_cond is not None else torch.zeros(size=(77, 768), dtype=torch.float32, device=shared.device)

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

    def build(self, conds):
        max_cond_size = self.__max_cond_size(conds)
        conds = self.__resize_uniformly(conds, max_cond_size)
        return InterpolationTensor(
            InterpolationTensorBuilder.__build_conditionings_tensor(self.__indices_tensor, conds),
            self.__interpolation_functions,
            self.__empty_cond)

    @staticmethod
    def __build_conditionings_tensor(tensor, conds):
        if type(tensor) is int:
            return conds[tensor]
        else:
            return [InterpolationTensorBuilder.__build_conditionings_tensor(e, conds) for e in tensor]

    def __resize_uniformly(self, conds, max_cond_size: int):
        conds[:] = ([self.__resize_schedule(schedule, max_cond_size) for schedule in schedules]
                    for schedules in conds)
        return conds

    def __resize_schedule(self, schedule, target_size):
        cond_missing_size = (target_size - schedule.cond.size(0)) // 77
        if cond_missing_size <= 0:
            return schedule

        resized_cond = torch.concatenate([schedule.cond] + [self.__empty_cond] * cond_missing_size)
        return prompt_parser.ScheduledPromptConditioning(cond=resized_cond, end_at_step=schedule.end_at_step)

    @staticmethod
    def __max_cond_size(conds):
        return max(schedule.cond.size(0)
                   for schedules in conds
                   for schedule in schedules)
