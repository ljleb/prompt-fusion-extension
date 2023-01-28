class InterpolationTensor:
    def __init__(self, conditionings_tensor, interpolation_functions):
        self.__conditionings_tensor = conditionings_tensor
        self.__interpolation_functions = interpolation_functions

    def interpolate(self, t, step, axis=0):
        tensor_axes = len(self.__interpolation_functions) - axis
        if tensor_axes == 0:
            if type(self.__conditionings_tensor) is not list:
                return self.__conditionings_tensor

            for schedule in self.__conditionings_tensor:
                if schedule.end_at_step >= step:
                    return schedule.cond

        interpolation_function, control_points_functions = self.__interpolation_functions[axis]
        if tensor_axes == 1:
            control_points = list(self.__conditionings_tensor)
        else:
            control_points = [InterpolationTensor(sub_tensor, self.__interpolation_functions).interpolate(t, step, axis + 1)
                              for sub_tensor in self.__conditionings_tensor]

        for i, nested_functions in enumerate(control_points_functions):
            control_points[i] = InterpolationTensor(control_points[i], nested_functions).interpolate(t, step)

        return interpolation_function(t, control_points)


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

    def build(self, conditionings):
        return InterpolationTensor(
            InterpolationTensorBuilder.__build_conditionings_tensor(self.__indices_tensor, conditionings),
            self.__interpolation_functions)

    @staticmethod
    def __build_conditionings_tensor(tensor, conditionings):
        if type(tensor) is int:
            return conditionings[tensor]
        else:
            return [InterpolationTensorBuilder.__build_conditionings_tensor(e, conditionings) for e in tensor]
