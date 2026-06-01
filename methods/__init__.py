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
    # Math-solving persona set (strategize -> compute -> verify). 'verify' is a
    # text-producer like 'judger' (see TEXT_PRODUCER_ROLES in latent_mas.py): it
    # emits the final \boxed answer; strategize/compute do latent steps only.
    "strategize": "Strategist",
    "compute": "Calculator",
    "verify": "Verifier",
}

# Closed set: each role must have a prompt template in prompts.py. Adding a
# new role requires extending build_agent_message_sequential_latent_mas etc.
KNOWN_ROLES = frozenset(_DEFAULT_NAMES.keys())

# Roles that produce the final TEXT answer rather than feeding the next agent.
# In latent_mas they decode with the prior agents' latent KV as context; in
# text_mas they set final_texts instead of appending to the running context.
# The last agent in a pipeline should be one of these.
TEXT_PRODUCER_ROLES = ("judger", "verify")


def _make_agent(role: str) -> Agent:
    role = role.strip().lower()
    if role not in KNOWN_ROLES:
        raise ValueError(
            f"Unknown agent role '{role}'. Known roles: {sorted(KNOWN_ROLES)}. "
            f"To add a new role, also add a prompt branch in prompts.py."
        )
    return Agent(name=_DEFAULT_NAMES[role], role=role)


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


__all__ = ["Agent", "Parallel", "PipelineOp", "default_agents", "parse_pipeline", "KNOWN_ROLES", "TEXT_PRODUCER_ROLES"]
