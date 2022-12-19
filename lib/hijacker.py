from modules import prompt_parser


class ModuleHijacker:
    def __init__(self, module):
        self.__module = module
        self.__original_functions = dict()

    def hijack(self, attribute):
        if attribute not in self.__original_functions:
            self.__original_functions[attribute] = getattr(self.__module, attribute)

        def decorator(function):
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs, original_function=self.__original_functions[attribute])

            setattr(self.__module, attribute, wrapper)
            return function

        return decorator

    @staticmethod
    def install_or_get(module, hijacker_attribute):
        if not hasattr(module, hijacker_attribute):
            module_hijacker = ModuleHijacker(module)
            setattr(module, hijacker_attribute, module_hijacker)
            return module_hijacker
        else:
            return getattr(module, hijacker_attribute)


fusion_hijacker_attribute = '__fusion_hijacker'
prompt_parser_hijacker = ModuleHijacker.install_or_get(prompt_parser, fusion_hijacker_attribute)
