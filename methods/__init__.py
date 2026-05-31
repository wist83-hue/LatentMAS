import re
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Agent:
    name: str
    role: str


@dataclass
class Parallel:
    branches: List[List["Agent"]] = field(default_factory=list)


PipelineOp = Union[Agent, Parallel]


_DEFAULT_NAMES = {
    "planner": "Planner",
    "critic": "Critic",
    "refiner": "Refiner",
    "judger": "Judger",
}


def _make_agent(role: str) -> Agent:
    role = role.strip().lower()
    return Agent(name=_DEFAULT_NAMES.get(role, role.capitalize()), role=role)


def _split_top_level(spec: str, sep: str) -> List[str]:
    """Split spec on `sep` at top level (depth 0), respecting parens."""
    out: List[str] = []
    depth = 0
    current: List[str] = []
    for c in spec:
        if c == "(":
            depth += 1
            current.append(c)
        elif c == ")":
            depth -= 1
            current.append(c)
        elif c == sep and depth == 0:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(c)
    if current:
        out.append("".join(current).strip())
    return [t for t in out if t]


def default_agents() -> List[Agent]:
    return [_make_agent(r) for r in ("planner", "critic", "refiner", "judger")]


def parse_pipeline(spec: str) -> List[PipelineOp]:
    out: List[PipelineOp] = []
    for tok in _split_top_level(spec, ","):
        # parallel(branch|branch|...)
        if tok.startswith("parallel(") and tok.endswith(")"):
            inner = tok[len("parallel("):-1]
            branch_specs = _split_top_level(inner, "|")
            branches = [parse_pipeline(b) for b in branch_specs]
            out.append(Parallel(branches=branches))
            continue
        # (role+role+...)*N loop
        m = re.match(r"^\(([^)]+)\)\*(\d+)$", tok)
        if m:
            roles = [r.strip().lower() for r in m.group(1).split("+")]
            times = int(m.group(2))
            for _ in range(times):
                for role in roles:
                    out.append(_make_agent(role))
            continue
        # bare role
        out.append(_make_agent(tok))
    return out


__all__ = ["Agent", "Parallel", "PipelineOp", "default_agents", "parse_pipeline"]
