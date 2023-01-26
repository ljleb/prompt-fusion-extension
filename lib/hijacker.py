from modules import prompt_parser
from modules.script_callbacks import on_script_unloaded


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

    def reset_module(self):
        for attribute, original_function in self.__original_functions.items():
            setattr(self.__module, attribute, original_function)

    @staticmethod
    def install_or_get(module, hijacker_attribute, register_uninstall=lambda _callback: None):
        if not hasattr(module, hijacker_attribute):
            module_hijacker = ModuleHijacker(module)
            setattr(module, hijacker_attribute, module_hijacker)
            register_uninstall(lambda: delattr(prompt_parser, hijacker_attribute))
            register_uninstall(module_hijacker.reset_module)
            return module_hijacker
        else:
            return getattr(module, hijacker_attribute)


prompt_parser_hijacker = None


def prepare_hijack():
    global prompt_parser_hijacker
    fusion_hijacker_attribute = '__fusion_hijacker'
    prompt_parser_hijacker = ModuleHijacker.install_or_get(prompt_parser, fusion_hijacker_attribute, on_script_unloaded)
