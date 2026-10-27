"""Unit tests for `RngTree`: one seed, one stream for each ask and component."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from planexs.core.ConfigError import ConfigError
from planexs.core.RngTree import RngTree

#: Streams, as (seed, ask index, path) and the integer seed derived for each. These pin the
#: derivation: it cannot change without breaking every saved study (ADR-0011), so a change that
#: moves these numbers is a deliberate break, to be listed in the changelog.
KNOWN_SEEDS = [
    (0, 0, "", 16736464761881911207),
    (7, 0, "policy", 6459392208546183312),
    (7, 3, "policy.surrogate.kernel", 8744225987445234153),
    (2**100 + 5, 10, "acquisition_optimizer", 16534859845684878730),
    (123, 2, "design", 14160748161884979063),
]

#: The first two raw outputs of the `PCG64` generator of some of those streams.
KNOWN_RAW = [
    (0, 0, "", [5832345686816349673, 784102599806925178]),
    (7, 0, "policy", [6575831566151537518, 3536550039925009771]),
    (7, 3, "policy.surrogate.kernel", [14217021162287590816, 9693722952894489384]),
]

PATHS = ["policy", "policy.design", "policy.surrogate", "policy.surrogate.kernel", "analysis"]


def draws(generator: Any, count: int = 8) -> list[float]:
    """The first `count` numbers a generator produces."""
    return [float(value) for value in generator.random(count)]


def failureOf(call: Any, *arguments: Any) -> ConfigError:
    """The `ConfigError` that `call(*arguments)` raises."""
    with pytest.raises(ConfigError) as caught:
        call(*arguments)
    return caught.value


class TestTheSameInputsGiveTheSameStream:
    def test_the_same_arguments_give_the_same_integer_seed(self) -> None:
        tree = RngTree(7)
        assert tree.seed(3, "policy") == tree.seed(3, "policy")

    def test_the_same_arguments_give_generators_that_produce_the_same_numbers(self) -> None:
        tree = RngTree(7)
        assert draws(tree.generator(3, "policy")) == draws(tree.generator(3, "policy"))

    def test_two_trees_of_the_same_seed_have_the_same_streams(self) -> None:
        first, second = RngTree(7), RngTree(7)
        for ask in range(4):
            for path in PATHS:
                assert first.seed(ask, path) == second.seed(ask, path)
                assert draws(first.generator(ask, path)) == draws(second.generator(ask, path))

    def test_every_call_returns_a_new_generator_at_the_start_of_its_stream(self) -> None:
        tree = RngTree(7)
        first, second = tree.generator(0, "policy"), tree.generator(0, "policy")
        assert first is not second
        first.random(100)  # using one does not move the other
        assert draws(second) == draws(tree.generator(0, "policy"))

    def test_the_generator_is_over_the_pcg64_bit_generator(self) -> None:
        assert isinstance(RngTree(7).generator(0, "policy"), np.random.Generator)
        assert isinstance(RngTree(7).generator(0, "policy").bit_generator, np.random.PCG64)

    def test_a_seed_of_any_size_works(self) -> None:
        assert RngTree(2**200).seed(1, "x") == RngTree(2**200).seed(1, "x")
        assert RngTree(2**200).seed(1, "x") != RngTree(2**200 + 1).seed(1, "x")

    def test_an_integer_seed_is_a_python_int_below_2_to_the_64(self) -> None:
        for _, ask, path, _ in KNOWN_SEEDS:
            seed = RngTree(7).seed(ask, path)
            assert type(seed) is int
            assert 0 <= seed < 2**64

    def test_an_integer_seed_makes_a_torch_style_generator_reproducible(self) -> None:
        # What a seed is for: a library that takes an integer. A numpy generator stands in
        # for one here (torch is not imported by the core).
        seed = RngTree(7).seed(0, "policy")
        assert draws(np.random.default_rng(seed)) == draws(np.random.default_rng(seed))


class TestDifferentInputsGiveDifferentStreams:
    def test_another_ask_index_gives_another_stream(self) -> None:
        tree = RngTree(7)
        assert tree.seed(0, "policy") != tree.seed(1, "policy")
        assert draws(tree.generator(0, "policy")) != draws(tree.generator(1, "policy"))

    def test_another_path_gives_another_stream(self) -> None:
        tree = RngTree(7)
        assert tree.seed(0, "policy.design") != tree.seed(0, "policy.surrogate")
        assert draws(tree.generator(0, "a")) != draws(tree.generator(0, "b"))

    def test_another_seed_gives_another_stream(self) -> None:
        assert RngTree(7).seed(0, "policy") != RngTree(8).seed(0, "policy")
        assert draws(RngTree(7).generator(0, "p")) != draws(RngTree(8).generator(0, "p"))

    def test_every_stream_of_a_grid_of_asks_and_paths_is_different(self) -> None:
        tree = RngTree(7)
        seeds = {(ask, path): tree.seed(ask, path) for ask in range(10) for path in PATHS}
        assert len(set(seeds.values())) == len(seeds)

    @pytest.mark.parametrize(
        "other", ["policy.", "Policy", " policy", "policy ", "polícy", ""]
    )
    def test_paths_that_look_alike_are_different_paths(self, other: str) -> None:
        tree = RngTree(7)
        assert tree.seed(0, "policy") != tree.seed(0, other)

    def test_the_ask_index_and_the_path_are_not_interchangeable(self) -> None:
        tree = RngTree(7)
        assert tree.seed(1, "2") != tree.seed(2, "1")

    def test_the_ask_index_is_not_a_shift_of_the_stream(self) -> None:
        # The stream of ask 1 is not the stream of ask 0 started a number later.
        tree = RngTree(7)
        zero = draws(tree.generator(0, "policy"), 20)
        one = draws(tree.generator(1, "policy"), 20)
        assert set(zero).isdisjoint(one)


class TestOtherStreamsDoNotMatter:
    """Adding a component, or asking in another order, does not change a stream."""

    def test_asking_for_unrelated_paths_does_not_change_a_stream(self) -> None:
        tree = RngTree(7)
        before = (tree.seed(0, "a"), draws(tree.generator(0, "a")))
        for ask in range(5):
            for path in ("b", "c", "d", "policy.new_component", "e.f.g"):
                tree.seed(ask, path)
                tree.generator(ask, path).random(50)
        assert (tree.seed(0, "a"), draws(tree.generator(0, "a"))) == before

    def test_a_tree_that_asked_for_other_paths_equals_one_that_never_did(self) -> None:
        used, fresh = RngTree(7), RngTree(7)
        for path in ("b", "c", "d"):
            used.generator(0, path).random(10)
        assert draws(used.generator(0, "a")) == draws(fresh.generator(0, "a"))

    def test_the_order_in_which_streams_are_asked_for_does_not_matter(self) -> None:
        forward, backward = RngTree(7), RngTree(7)
        ordered = [(ask, path) for ask in range(3) for path in PATHS]
        got_forward = {key: forward.seed(*key) for key in ordered}
        got_backward = {key: backward.seed(*key) for key in reversed(ordered)}
        assert got_forward == got_backward

    def test_using_one_stream_does_not_change_another(self) -> None:
        tree = RngTree(7)
        expected = draws(RngTree(7).generator(0, "quiet"))
        busy = tree.generator(0, "busy")
        busy.random(10_000)
        assert draws(tree.generator(0, "quiet")) == expected

    def test_a_component_added_to_a_study_leaves_the_others_as_they_were(self) -> None:
        # The streams of a study of three components, then of the same study with a fourth
        # one whose path sorts between them: the first three are unchanged.
        paths = ["policy.design", "policy.surrogate", "policy.zzz"]
        tree = RngTree(7)
        before = {path: tree.seed(4, path) for path in paths}
        tree.seed(4, "policy.mmm")  # the new component
        assert {path: tree.seed(4, path) for path in paths} == before


class TestTheRecordedEntropy:
    def test_a_given_seed_is_its_own_entropy(self) -> None:
        assert RngTree(7).entropy == 7
        assert RngTree(0).entropy == 0
        assert RngTree(2**100).entropy == 2**100

    def test_entropy_is_a_python_int(self) -> None:
        assert type(RngTree(7).entropy) is int
        assert type(RngTree().entropy) is int

    def test_without_a_seed_entropy_is_drawn_from_the_operating_system(self) -> None:
        first, second = RngTree(), RngTree()
        assert first.entropy != second.entropy
        assert first.entropy.bit_length() > 64  # a wide number, not a small default
        assert first.seed(0, "policy") != second.seed(0, "policy")

    def test_a_tree_rebuilt_from_the_recorded_entropy_has_the_same_streams(self) -> None:
        original = RngTree()
        rebuilt = RngTree(original.entropy)
        assert rebuilt.entropy == original.entropy
        for ask in range(3):
            for path in PATHS:
                assert rebuilt.seed(ask, path) == original.seed(ask, path)
                assert draws(rebuilt.generator(ask, path)) == draws(
                    original.generator(ask, path)
                )

    def test_the_entropy_survives_json(self) -> None:
        # A snapshot is JSON (specification 5.11).
        original = RngTree()
        recorded = json.loads(json.dumps({"entropy": original.entropy}))["entropy"]
        assert RngTree(recorded).seed(2, "policy") == original.seed(2, "policy")

    def test_the_entropy_does_not_change_as_streams_are_used(self) -> None:
        tree = RngTree()
        entropy = tree.entropy
        tree.generator(0, "policy").random(100)
        assert tree.entropy == entropy


class TestNoGlobalRandomState:
    def test_the_global_states_are_left_alone(self) -> None:
        np.random.seed(2024)
        random.seed(2024)
        numpy_before, python_before = np.random.get_state(), random.getstate()
        tree = RngTree()
        for ask in range(3):
            tree.seed(ask, "policy")
            tree.generator(ask, "policy").random(20)
        numpy_after = np.random.get_state()
        assert numpy_after[0] == numpy_before[0]
        assert (numpy_after[1] == numpy_before[1]).all()
        assert numpy_after[2:] == numpy_before[2:]
        assert random.getstate() == python_before

    def test_the_streams_do_not_depend_on_the_global_states(self) -> None:
        np.random.seed(1)
        random.seed(1)
        first = (RngTree(7).seed(0, "p"), draws(RngTree(7).generator(0, "p")))
        np.random.seed(999)
        random.seed(999)
        second = (RngTree(7).seed(0, "p"), draws(RngTree(7).generator(0, "p")))
        assert first == second


class TestTheDerivationIsTheDocumentedOne:
    @staticmethod
    def documented(seed: int, ask_index: int, path: str) -> Any:
        """The stream's `SeedSequence`, computed as the docstring of `RngTree` says."""
        digest = hashlib.sha256(path.encode("utf-8")).digest()
        key = (ask_index, int.from_bytes(digest, "big"))
        return np.random.SeedSequence(entropy=seed, spawn_key=key)

    @pytest.mark.parametrize("seed, ask, path, expected", KNOWN_SEEDS)
    def test_the_integer_seed_is_the_first_64_bit_word_of_the_sequence(
        self, seed: int, ask: int, path: str, expected: int
    ) -> None:
        word = self.documented(seed, ask, path).generate_state(1, np.uint64)[0]
        assert RngTree(seed).seed(ask, path) == int(word)

    @pytest.mark.parametrize("seed, ask, path, expected", KNOWN_SEEDS)
    def test_the_generator_is_pcg64_over_that_sequence(
        self, seed: int, ask: int, path: str, expected: int
    ) -> None:
        reference = np.random.Generator(np.random.PCG64(self.documented(seed, ask, path)))
        assert draws(RngTree(seed).generator(ask, path)) == draws(reference)

    @pytest.mark.parametrize("seed, ask, path, expected", KNOWN_SEEDS)
    def test_known_seeds(self, seed: int, ask: int, path: str, expected: int) -> None:
        assert RngTree(seed).seed(ask, path) == expected

    @pytest.mark.parametrize("seed, ask, path, expected", KNOWN_RAW)
    def test_known_raw_output_of_the_generators(
        self, seed: int, ask: int, path: str, expected: list[int]
    ) -> None:
        raw = RngTree(seed).generator(ask, path).bit_generator.random_raw(2)
        assert [int(value) for value in raw] == expected

    def test_the_path_is_hashed_with_sha256_and_not_with_python_hash(self) -> None:
        # `hash` of a string differs from one process to the next: the next class proves the
        # streams do not, and this one that the derivation uses the digest of the path.
        digest = hashlib.sha256(b"policy").digest()
        key = (0, int.from_bytes(digest, "big"))
        reference = np.random.SeedSequence(entropy=7, spawn_key=key)
        assert RngTree(7).seed(0, "policy") == int(reference.generate_state(1, np.uint64)[0])

    def test_the_path_is_read_as_utf_8(self) -> None:
        digest = hashlib.sha256("é☃".encode()).digest()
        reference = np.random.SeedSequence(
            entropy=7, spawn_key=(0, int.from_bytes(digest, "big"))
        )
        assert RngTree(7).seed(0, "é☃") == int(reference.generate_state(1, np.uint64)[0])


#: Run in another process: it loads only `planexs.core` (a stand-in for the package keeps the
#: heavy libraries that `planexs/__init__.py` imports out of it), so it also shows that the
#: tree needs none of them. It prints the streams of a tree as JSON.
SCRIPT = """
import json, sys, types
source = sys.argv[1]
package = types.ModuleType("planexs")
package.__path__ = [source + "/planexs"]
sys.modules["planexs"] = package
from planexs.core.RngTree import RngTree

tree = RngTree(int(sys.argv[2]))
streams = {}
for ask in range(3):
    for path in ["policy", "policy.surrogate.kernel", "analysis", ""]:
        raw = tree.generator(ask, path).bit_generator.random_raw(3)
        streams[f"{ask}|{path}"] = [tree.seed(ask, path), [int(value) for value in raw]]
print(json.dumps({
    "streams": streams,
    "hash_seed": sys.flags.hash_randomization,
    "heavy": [name for name in ("torch", "botorch", "gpytorch", "ax") if name in sys.modules],
}))
"""


class TestAcrossProcesses:
    """The same seed gives the same streams in another process, whatever Python's hash seed."""

    @staticmethod
    def streamsInOtherProcess(hash_seed: str, seed: int) -> dict[str, Any]:
        source = str(Path(inspect.getfile(RngTree)).resolve().parents[2])
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT, source, str(seed)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        parsed: dict[str, Any] = json.loads(result.stdout)
        return parsed

    @staticmethod
    def streamsHere(seed: int) -> dict[str, Any]:
        tree = RngTree(seed)
        streams: dict[str, Any] = {}
        for ask in range(3):
            for path in ["policy", "policy.surrogate.kernel", "analysis", ""]:
                raw = tree.generator(ask, path).bit_generator.random_raw(3)
                streams[f"{ask}|{path}"] = [
                    tree.seed(ask, path),
                    [int(value) for value in raw],
                ]
        return streams

    @pytest.mark.parametrize("hash_seed", ["0", "1", "424242", "random"])
    def test_another_process_derives_the_same_streams(self, hash_seed: str) -> None:
        other = self.streamsInOtherProcess(hash_seed, 7)
        assert other["streams"] == self.streamsHere(7)

    def test_two_processes_with_different_hash_seeds_agree_with_each_other(self) -> None:
        first = self.streamsInOtherProcess("1", 12345)
        second = self.streamsInOtherProcess("2", 12345)
        assert first["streams"] == second["streams"]
        assert first["streams"] != self.streamsInOtherProcess("1", 12346)["streams"]

    def test_the_other_process_really_ran_with_a_hash_seed_of_its_own(self) -> None:
        # Python's `hash` of a string would have differed between these two runs.
        probe = "import sys; print(hash('planexs'))"
        values = {
            subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
            ).stdout
            for hash_seed in ("1", "2")
        }
        assert len(values) == 2

    def test_the_tree_needs_none_of_the_heavy_libraries(self) -> None:
        assert self.streamsInOtherProcess("0", 7)["heavy"] == []


class TestWhatIsNotAValidArgument:
    @pytest.mark.parametrize("seed", [-1, -(2**70)])
    def test_a_negative_seed(self, seed: int) -> None:
        error = failureOf(RngTree, seed)
        assert error.field_path == "seed"
        assert error.message == (
            f"The seed of a random-number tree must be a non-negative integer; got {seed!r}."
        )
        assert error.suggestion == "give a whole number that is zero or more"

    @pytest.mark.parametrize("seed", [1.5, "7", [7], True, False, float("nan")], ids=repr)
    def test_a_seed_that_is_not_a_whole_number(self, seed: object) -> None:
        assert failureOf(RngTree, seed).field_path == "seed"

    def test_a_numpy_integer_is_a_whole_number(self) -> None:
        tree = RngTree(np.int64(7))  # type: ignore[arg-type]
        assert tree.entropy == 7
        assert type(tree.entropy) is int
        assert tree.seed(np.int64(3), "p") == RngTree(7).seed(3, "p")  # type: ignore[arg-type]

    def test_zero_is_a_seed_and_an_ask_index(self) -> None:
        assert RngTree(0).seed(0, "p") == RngTree(0).seed(0, "p")

    @pytest.mark.parametrize("ask", [-1, 1.0, "0", None, True, [0]], ids=repr)
    def test_an_ask_index_that_is_not_a_non_negative_integer(self, ask: Any) -> None:
        tree = RngTree(7)
        assert failureOf(tree.seed, ask, "p").field_path == "ask_index"
        assert failureOf(tree.generator, ask, "p").field_path == "ask_index"

    @pytest.mark.parametrize("path", [None, 3, b"policy", ("policy",), ["policy"]], ids=repr)
    def test_a_path_that_is_not_a_string(self, path: Any) -> None:
        tree = RngTree(7)
        error = failureOf(tree.seed, 0, path)
        assert error.field_path == "path"
        assert error.message == (
            f"The path of a stream must be a string; got {type(path).__name__}."
        )
        assert failureOf(tree.generator, 0, path).field_path == "path"

    def test_an_empty_path_is_a_path(self) -> None:
        tree = RngTree(7)
        assert tree.seed(0, "") == tree.seed(0, "")
        assert tree.seed(0, "") != tree.seed(0, "policy")

    def test_a_wrong_argument_is_a_config_error_and_not_a_numpy_error(self) -> None:
        # numpy's own `ValueError` and `TypeError` never reach the caller.
        with pytest.raises(ConfigError):
            RngTree(-5)
        with pytest.raises(ConfigError):
            RngTree(1.5)  # type: ignore[arg-type]
