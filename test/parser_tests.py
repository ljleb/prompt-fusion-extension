from lib_prompt_fusion.prompt_parser import parse_prompt
from lib_prompt_fusion.interpolation_tensor import InterpolationTensorBuilder


def run_functional_tests(total_steps=100):
    for i, (given, expected) in enumerate(functional_parse_test_cases):
        expr = parse_prompt(given)
        tensor_builder = InterpolationTensorBuilder()
        expr.extend_tensor(tensor_builder, (0, total_steps), total_steps, dict())

        actual = tensor_builder.get_prompt_database()

        if type(expected) is set:
            assert set(actual) == expected, f"{actual} != {expected}"
        else:
            assert len(actual) == 1 and actual[0] == expected, f"'{actual[0]}' != '{expected}'"


functional_parse_test_cases = [
    ('single',)*2,
    ('some space separated text',)*2,
    ('(legacy weighted prompt:-2.1)',)*2,
    ('mixed (legacy weight:3.6) and text',)*2,
    ('legacy [range begin:0] thingy',)*2,
    ('legacy [range end::3] thingy',)*2,
    ('legacy [[nested range::3]:2] thingy',)*2,
    ('legacy [[nested range:2]::3] thingy',)*2,
    ('sugar [range:,abc:3] thingy',)*2,
    ('sugar [[(weight interpolation:0,12):0]::1] thingy', 'sugar [[(weight interpolation:0.0):0]::1] thingy'),
    ('sugar [[(weight interpolation:0,12):0]::2] thingy', 'sugar [[[(weight interpolation:0.0)::1][(weight interpolation:12.0):1]:0]::2] thingy'),
    ('sugar [[(weight interpolation:0,12):0]::3] thingy', 'sugar [[[(weight interpolation:0.0)::1][[(weight interpolation:6.0):1]::2][(weight interpolation:12.0):2]:0]::3] thingy'),
    ('legacy [from:to:2] thingy',)*2,
    ('legacy [negative weight]',)*2,
    ('legacy (positive weight)',)*2,
    ('[abc:1girl:2]',)*2,
    ('1girl',)*2,
    ('dashes-in-text',)*2,
    ('text, separated with, comas',)*2,
    ('{prompt}',)*2,
    ('[abc|def ghi|jkl]',)*2,
    ('merging this AND with this',)*2,
    (':',)*2,
    (r'portrait \(object\)',)*2,
    (r'\[escaped square\]',)*2,
    (r'\$var = abc',)*2,
    ('$var = abc', ''),
    (r'\\$ arst',)*2,
    (r'$$ arst',)*2,
    ('$a = prompt value\n$a', 'prompt value'),
    ('$a = prompt value\n$b = $a\n$b', 'prompt value'),
    ('$a = (multiline\nprompt\nvalue:1.0)\n$a', '(multiline prompt value:1.0)'),
    ('$a = ($aa = nested variable\nmultiline\n$aa:1.0)\n$a', '(multiline nested variable:1.0)'),
    ('a [b:c:-1, 10] d', {'a b d', 'a c d'}),
    ('a [b:c:5, 6] d', {'a b d', 'a c d'}),
    ('a [b:c:0.25, 0.5] d', {'a b d', 'a c d'}),
    ('a [b:c:.25, .5] d', {'a b d', 'a c d'}),
    ('a [b:c:,] d', {'a b d', 'a c d'}),
    ('0[1.0:1.1:,]2[3.0:3.1:,]4', {
        '0 1.0 2 3.0 4', '0 1.1 2 3.0 4',
        '0 1.0 2 3.1 4', '0 1.1 2 3.1 4',
    }),
    ('0[1.0:1.1:1.2:,.5,]2[3.0:3.1:,]4', {
        '0 1.0 2 3.0 4', '0 1.0 2 3.1 4',
        '0 1.1 2 3.0 4', '0 1.1 2 3.1 4',
        '0 1.2 2 3.0 4', '0 1.2 2 3.1 4',
    }),
    ('[0.0:0.1:,][1.0:1.1:,][2.0:2.1:,]', {
        '0.0 1.0 2.0', '0.0 1.0 2.1',
        '0.1 1.0 2.0', '0.1 1.0 2.1',
        '0.0 1.1 2.0', '0.0 1.1 2.1',
        '0.1 1.1 2.0', '0.1 1.1 2.1',
    }),
    ('[top level:interpolatin:lik a pro:1,3,5:linear]', {'top level', 'interpolatin', 'lik a pro'}),
    ('[[nested:expr:,]:abc:,]', {'nested', 'expr', 'abc'}),
    ('[(nested attention:2.0):abc:,]', {'(nested attention:2.0)', 'abc'}),
    ('[[nested editing:15]:abc:,]', {'[nested editing:15]', 'abc'}),
    ('[[nested interpolation:abc:,]:12]', {'[nested interpolation:12]', '[abc:12]'}),
    ('[[nested interpolation:abc:,]::7]', {'[nested interpolation::7]', '[abc::7]'}),
    ('$attention = 1.5\n(prompt:$attention)', '(prompt:1.5)'),
    ('$a = 0\n$b = 12\n[[(prompt:$a,$b):0]::2]', '[[[(prompt:0.0)::1][(prompt:12.0):1]:0]::2]'),
    ('$step = 5\n[legacy:editing:$step]', '[legacy:editing:5]'),
    ('$begin = 2\n$end = 7\n[prompt:interpolation:$begin, $end]', {'prompt', 'interpolation'}),
]


def run_tests():
    run_functional_tests()
