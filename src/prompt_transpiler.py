import re
from extensions.promptlang.src.dsl_prompt_transpiler import transpile_prompt as transpile_dsl
from extensions.promptlang.src.xml_prompt_transpiler import transpile_prompt as transpile_xml

header_re = re.compile("\(promptlang\)\[,0]")
header_xml_re = re.compile("<!--promptlang-->")


def transpile_prompts(prompts, steps):
    res = []

    for prompt in prompts:
        if header_re.match(prompt):
            res.append(transpile_dsl(prompt, steps))
        elif header_xml_re.match(prompt):
            res.append(transpile_xml(prompt, steps))
        else:
            res.append(prompt)

    return res
