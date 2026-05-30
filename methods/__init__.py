import re
from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


_DEFAULT_NAMES = {
    "planner": "Planner",
    "critic": "Critic",
    "refiner": "Refiner",
    "judger": "Judger",
}


def default_agents() -> List[Agent]:
    return [Agent(name=_DEFAULT_NAMES[r], role=r) for r in ("planner", "critic", "refiner", "judger")]


def parse_pipeline(spec: str) -> List[Agent]:
    out: List[Agent] = []
    for tok in spec.split(","):
        tok = tok.strip()
        m = re.match(r"^\(([^)]+)\)\*(\d+)$", tok)
        if m:
            roles = [r.strip().lower() for r in m.group(1).split("+")]
            times = int(m.group(2))
            for _ in range(times):
                for role in roles:
                    out.append(Agent(name=_DEFAULT_NAMES.get(role, role.capitalize()), role=role))
        else:
            role = tok.lower()
            out.append(Agent(name=_DEFAULT_NAMES.get(role, role.capitalize()), role=role))
    return out


__all__ = ["Agent", "default_agents", "parse_pipeline"]
