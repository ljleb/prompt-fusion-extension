import re
from lib.dsl_prompt_transpiler import transpile_prompt as transpile_dsl
from lib.xml_prompt_transpiler import transpile_prompt as transpile_xml

header_re = re.compile("\(fusion v0\)\[,0]")
header_xml_re = re.compile("<!--fusion v0-->")


def transpile_prompt(prompt, steps):
    if header_re.match(prompt):
        return transpile_dsl(prompt, steps)
    elif header_xml_re.match(prompt):
        return transpile_xml(prompt, steps)
    else:
        return prompt
