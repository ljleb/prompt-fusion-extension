import xml.etree.ElementTree as XmlParser
import lib.ast_nodes as ast


def transpile_prompt(prompt, steps):
    root = XmlParser.fromstring('<prompt>\n' + prompt + '\n</prompt>')
    expression = xml_to_ast(root, steps)
    return expression.evaluate((0, steps))


def xml_to_ast(node, steps):
    res = []

    def append_text(node_text):
        refined_text = (node_text or '').strip()
        if refined_text != '':
            res.append(ast.TextExpression(refined_text))

    append_text(node.text)

    for child in node:
        if child.tag == 'range':
            start = int(child.get('start') or '0')
            end = int(child.get('end') or str(steps))
            res.append(ast.RangeExpression(xml_to_ast(child, steps), start, end))

        if child.tag == 'weight':
            weight = float(child.get('value') or '1')
            res.append(ast.WeightedExpression(xml_to_ast(child, steps), weight))

        if child.tag == 'weight-interpolation':
            start = float(child.get('start') or '1')
            end = float(child.get('end') or '1')
            res.append(ast.WeightInterpolationExpression(xml_to_ast(child, steps), start, end))

        append_text(child.tail)

        xml_to_ast(child, steps)
    return ast.ListExpression(res)
