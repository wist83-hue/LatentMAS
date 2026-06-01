"""Tests for methods/__init__.py: Agent dataclass, parse_pipeline DSL, role validation."""
import pytest

from methods import (
    Agent,
    Parallel,
    KNOWN_ROLES,
    default_agents,
    parse_pipeline,
)


class TestKnownRoles:
    def test_default_set(self):
        assert KNOWN_ROLES == frozenset({
            "planner", "critic", "refiner", "judger",
            "strategize", "compute", "verify",
        })

    def test_math_persona_set_parses(self):
        out = parse_pipeline("strategize,compute,verify")
        assert [op.role for op in out] == ["strategize", "compute", "verify"]
        assert [op.name for op in out] == ["Strategist", "Calculator", "Verifier"]


class TestDefaultAgents:
    def test_returns_four_in_order(self):
        agents = default_agents()
        assert [a.role for a in agents] == ["planner", "critic", "refiner", "judger"]

    def test_names_capitalized(self):
        agents = default_agents()
        assert [a.name for a in agents] == ["Planner", "Critic", "Refiner", "Judger"]


class TestParsePipelineSequential:
    def test_simple(self):
        out = parse_pipeline("planner,critic,refiner,judger")
        assert len(out) == 4
        assert all(isinstance(op, Agent) for op in out)
        assert [op.role for op in out] == ["planner", "critic", "refiner", "judger"]

    def test_whitespace_tolerated(self):
        out = parse_pipeline("planner ,  critic , judger")
        assert [op.role for op in out] == ["planner", "critic", "judger"]

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="Unknown agent role"):
            parse_pipeline("planner,solver,judger")

    def test_case_insensitive(self):
        out = parse_pipeline("PLANNER,Judger")
        assert [op.role for op in out] == ["planner", "judger"]


class TestParsePipelineLoop:
    def test_simple_loop(self):
        out = parse_pipeline("planner,(critic+refiner)*2,judger")
        # planner | critic refiner critic refiner | judger
        assert [op.role for op in out] == [
            "planner", "critic", "refiner", "critic", "refiner", "judger",
        ]

    def test_loop_once(self):
        out = parse_pipeline("(planner+critic+refiner)*1,judger")
        assert [op.role for op in out] == ["planner", "critic", "refiner", "judger"]

    def test_loop_zero_times(self):
        out = parse_pipeline("(planner+critic)*0,judger")
        assert [op.role for op in out] == ["judger"]

    def test_loop_unknown_role(self):
        with pytest.raises(ValueError):
            parse_pipeline("(planner+nonsense)*2,judger")


class TestParsePipelineParallel:
    def test_two_branches(self):
        out = parse_pipeline("parallel(planner|critic),judger")
        assert len(out) == 2
        assert isinstance(out[0], Parallel)
        assert isinstance(out[1], Agent)
        assert len(out[0].branches) == 2
        # Each branch is parse_pipeline-recursed -> List[Agent]
        assert out[0].branches[0][0].role == "planner"
        assert out[0].branches[1][0].role == "critic"
        assert out[1].role == "judger"

    def test_three_branches(self):
        out = parse_pipeline("parallel(planner|critic|refiner),judger")
        par = out[0]
        assert len(par.branches) == 3
        assert [b[0].role for b in par.branches] == ["planner", "critic", "refiner"]

    def test_parallel_unknown_role(self):
        with pytest.raises(ValueError):
            parse_pipeline("parallel(planner|nonsense),judger")


class TestParenAwareTokenizer:
    def test_comma_inside_parallel_does_not_split(self):
        # If we naively split on top-level comma, "parallel(a|b),judger" would
        # parse as expected, but a buggy parser might split on the | as well.
        # Test for explicit non-splitting on inner |.
        out = parse_pipeline("parallel(planner|critic),judger")
        assert len(out) == 2  # not 3

    def test_nested_loop_inside_parallel_branch(self):
        # A branch can itself contain a sequential subpipeline, including a loop
        out = parse_pipeline("parallel(planner|(critic+refiner)*2),judger")
        par = out[0]
        # branch 0: [planner]
        assert [a.role for a in par.branches[0]] == ["planner"]
        # branch 1: [critic, refiner, critic, refiner]
        assert [a.role for a in par.branches[1]] == ["critic", "refiner", "critic", "refiner"]


class TestAgentDataclass:
    def test_equality(self):
        a = Agent(name="Planner", role="planner")
        b = Agent(name="Planner", role="planner")
        assert a == b

    def test_inequality(self):
        a = Agent(name="Planner", role="planner")
        b = Agent(name="Critic", role="critic")
        assert a != b
