"""Fail if Python files use syntax or stdlib APIs introduced after Python 3.8.

The checker itself is included in the scan set and must remain Python 3.8-clean.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set

BANNED_ATTRS = {
    "asyncio": {"to_thread"},
    "functools": {"cache"},
    "str": {"removeprefix", "removesuffix"},
    "math": {"lcm", "nextafter"},
    "int": {"bit_count"},
}
PEP585_NAMES = {"list", "dict", "set", "tuple", "frozenset", "type"}
BUILTIN_TYPES = {
    "int", "str", "bytes", "float", "bool", "complex",
    "list", "dict", "set", "tuple", "frozenset", "type", "object",
}


def _looks_like_type(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id[:1].isupper() or node.id in BUILTIN_TYPES
    if isinstance(node, ast.Attribute):
        return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    return False


class Visitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: List[str] = []

    def error(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append("{}:{}: {}".format(self.path, line, message))

    def visit_Match(self, node: ast.AST) -> None:
        self.error(node, "match statement requires Python 3.10")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in PEP585_NAMES:
            self.error(node, "PEP 585 built-in generic '{}[...]' is not py38-safe".format(node.value.id))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr) and (
            _looks_like_type(node.left) or _looks_like_type(node.right)
        ):
            self.error(node, "possible PEP 604 union 'X | Y' is not py38-safe")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            module = node.value.id
            if node.attr in BANNED_ATTRS.get(module, set()):
                self.error(node, "{}.{} requires Python > 3.8".format(module, node.attr))
        self.generic_visit(node)


def check_file(path: Path) -> List[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except UnicodeDecodeError:
        return ["{}: source is not UTF-8".format(path)]
    except SyntaxError as exc:
        return ["{}:{}: syntax error: {}".format(path, exc.lineno, exc.msg)]
    visitor = Visitor(path)
    visitor.visit(tree)
    return visitor.errors


def main(argv: List[str]) -> int:
    roots = [Path(item) for item in argv[1:]] or [Path("src"), Path("scripts")]
    errors: List[str] = []
    seen: Set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() and root.suffix == ".py" else sorted(root.rglob("*.py"))
        for path in files:
            if path not in seen:
                seen.add(path)
                errors.extend(check_file(path))
    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        sys.stderr.write("\n{} Python 3.8 violation(s)\n".format(len(errors)))
        return 1
    sys.stdout.write("Python 3.8 AST check: OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
