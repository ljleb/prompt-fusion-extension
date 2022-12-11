import xml.etree.ElementTree as xml_parser
import extensions.promptlang.src.ast_nodes as ast


def transpile_prompt(prompt, steps):
    root = xml_parser.fromstring('<prompt>\n' + prompt + '\n</prompt>')
    expression = xml_to_ast(root, steps)
    return expression.evaluate((0, steps))


def xml_to_ast(node, steps):
    res = []
    
    text = (node.text or '').strip()
    if text != '':
        res.append(ast.TextExpression(text))

    for child in node:
        if child.tag == 'range':
            start = int(child.get('start') or '0')
            end = int(child.get('end') or str(steps))
            res.append(ast.RangeExpression(xml_to_ast(child, steps), start, end))

        if child.tag == 'weight':
            weight = float(child.get('value') or '1')
            res.append(ast.WeightedExpression(xml_to_ast(child, steps), weight))

        if child.tag == 'weight_interpolation':
            start = float(child.get('start') or '1')
            end = float(child.get('end') or '1')
            res.append(ast.WeightInterpolationExpression(xml_to_ast(child, steps), start, end))

        text = (child.tail or '').strip()
        if text != '':
            res.append(ast.TextExpression(text))
        xml_to_ast(child, steps)
    return ast.ListExpression(res)


if __name__ == '__main__':
    print(transpile_prompt('''
<weight value="1.2">
        extremely cute, beautiful and delicate 
        <range start="0" end="7">1girl, solo</range>
        <range start="7" end="30">loli girl</range>
</weight>,
caustics,

<weight value="0.8">
        frosty colors with frost
</weight>
<weight value="0.7">
        flashy sparkling red tunic with white stripes
</weight>
<weight value="1.2">
        extremely intricate spiky tattoos
</weight>
<weight value="1.1">
        very bright glowing yellow eyes
</weight>
<weight value="1.1">
        wearing a japanese hanbok
</weight>
''', 30))
