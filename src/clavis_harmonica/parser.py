import re
from typing import Dict, List
from .elements import ElementNode, build_element_node


def parse_formula(formula: str) -> List[ElementNode]:
    """Parses chemical formulas (including grouped parentheses like Ca(OH)2) into ElementNodes."""
    cleaned = formula.replace(" ", "")
    if not cleaned:
        raise ValueError("Formula cannot be empty.")

    tokens = re.findall(r"[A-Z][a-z]?\d*|\d+\.\d+|\d+|\(|\)", cleaned)
    stack: List[Dict[str, float]] = [{}]
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok == "(":
            stack.append({})
            i += 1
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError(f"Unmatched ')' in formula '{formula}'")
            group = stack.pop()
            i += 1
            mult = 1.0
            if i < len(tokens) and re.match(r"^\d+(\.\d+)?$", tokens[i]):
                mult = float(tokens[i])
                i += 1
            for el, cnt in group.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + cnt * mult
        elif re.match(r"^[A-Z][a-z]?\d*$", tok):
            el = tok
            i += 1
            mult = 1.0
            if i < len(tokens) and re.match(r"^\d+(\.\d+)?$", tokens[i]):
                mult = float(tokens[i])
                i += 1
            stack[-1][el] = stack[-1].get(el, 0.0) + mult
        else:
            raise ValueError(f"Unexpected token '{tok}' in formula '{formula}'")

    if len(stack) != 1:
        raise ValueError(f"Unclosed '(' in formula '{formula}'")

    return [build_element_node(sym, weight=w) for sym, w in stack[0].items()]
