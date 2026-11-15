"""Reject Python syntax newer than the project's Python 3.9 minimum.

PEP 585 built-in generics are valid on Python 3.9. PEP 604 unions and
structural pattern matching require Python 3.10 and are intentionally rejected.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set


class Visitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: List[str] = []

    def error(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append("{}:{}: {}".format(self.path, line, message))

    def visit_Match(self, node: ast.AST) -> None:
        self.error(node, "match statement requires Python 3.10")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr):
            self.error(node, "PEP 604 union 'X | Y' requires Python 3.10")
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
        sys.stderr.write("\n{} Python 3.9 violation(s)\n".format(len(errors)))
        return 1
    sys.stdout.write("Python 3.9 AST check: OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
