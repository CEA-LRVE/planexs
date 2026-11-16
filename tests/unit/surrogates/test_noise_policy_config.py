"""Unit tests for `NoisePolicyConfig` (roadmap task P3-08, specification 3.4, 5.4, 10.8).

The decision on partial knowledge of observation noise used to be a hidden rule of the legacy
`encodeTrials` (everything known or the noise is inferred). It is now one named setting, with
the rules `ignore_with_warning` (today's behaviour, and the default) and `error`.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.NoisePolicyConfig import NoisePolicyConfig, PartialNoiseRule


class TestTheSchema:
    def test_the_default_is_the_behaviour_the_library_had_before(self) -> None:
        assert NoisePolicyConfig().partial == "ignore_with_warning"

    def test_the_rules_are_exactly_the_two_of_the_card(self) -> None:
        assert set(get_args(PartialNoiseRule)) == {"ignore_with_warning", "error"}

    @pytest.mark.parametrize("rule", get_args(PartialNoiseRule))
    def test_every_rule_is_accepted_and_kept(self, rule: str) -> None:
        assert NoisePolicyConfig(partial=rule).partial == rule  # type: ignore[arg-type]

    def test_an_unknown_rule_is_refused_with_its_field_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(NoisePolicyConfig, {"partial": "average"})

        assert caught.value.field_path == "partial"
        assert "ignore_with_warning" in str(caught.value)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="parial"):
            NoisePolicyConfig.model_validate({"parial": "error"})

    def test_it_is_frozen(self) -> None:
        policy = NoisePolicyConfig()

        with pytest.raises(ValidationError, match="frozen"):
            policy.partial = "error"  # type: ignore[misc]

    def test_it_is_a_plain_options_schema_and_not_a_component(self) -> None:
        assert not issubclass(NoisePolicyConfig, ComponentConfig)
        assert "type" not in NoisePolicyConfig.model_fields


class TestItIsDocumented:
    def test_every_field_is_described_and_the_description_states_the_default(self) -> None:
        for name, info in NoisePolicyConfig.model_fields.items():
            assert info.description and info.description.strip(), name
            assert f"The default is '{info.default}'" in info.description, name

    @pytest.mark.parametrize("rule", get_args(PartialNoiseRule))
    def test_the_description_explains_every_rule(self, rule: str) -> None:
        description = NoisePolicyConfig.model_fields["partial"].description or ""

        assert f"'{rule}'" in description


class TestTheResolvedConfiguration:
    """A study stores its configuration with every default written out (specification 3.4)."""

    def test_the_default_is_written_out_in_full(self) -> None:
        assert toResolvedConfig(NoisePolicyConfig()) == {"partial": "ignore_with_warning"}

    @pytest.mark.parametrize("rule", get_args(PartialNoiseRule))
    def test_resolving_and_validating_again_gives_the_same_policy(self, rule: str) -> None:
        policy = NoisePolicyConfig(partial=rule)  # type: ignore[arg-type]

        resolved = toResolvedConfig(policy)

        assert validateConfig(NoisePolicyConfig, resolved) == policy
        assert toResolvedConfig(validateConfig(NoisePolicyConfig, resolved)) == resolved

    def test_a_configuration_that_leaves_the_field_out_takes_the_default(self) -> None:
        assert validateConfig(NoisePolicyConfig, {}) == NoisePolicyConfig()
