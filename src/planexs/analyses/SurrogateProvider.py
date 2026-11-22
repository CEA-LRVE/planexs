"""How an analysis obtains a surrogate, when it needs one."""

from __future__ import annotations

from typing import Protocol

from planexs.surrogates.Surrogate import Surrogate


class SurrogateProvider(Protocol):
    """What the study gives an analysis so that it can ask for a surrogate on demand
    (specification 5.10, 6).

    Fitting a surrogate costs far more than reading the history, and many analyses never need
    one (the best observed point, a Pareto front of the observed values), so an analysis is not
    given a surrogate: it is given a provider, and calls it only if it needs the model. The
    study fits at the first call and keeps the result (the surrogate cache is keyed by the
    history version and the resolved configuration, specification 5.6), so several analyses
    that ask for one in a row share a single fit.

    The provider is the study's own, and it is the same surrogate that the policy is using,
    never a different one (specification 5.13). Anything that is not a study can provide one
    too: a test passes a function that returns a fake, and a caller who only has a surrogate
    wraps it in a function that returns it. Any object with this method honours the protocol.
    """

    def surrogate(self) -> Surrogate:
        """The surrogate fitted to the history of the study the analysis is run on.

        It is a method, not a property, because the first call can fit a model
        (specification 10.2).

        Returns:
            The fitted surrogate, covering the outcomes that it models.

        Raises:
            Exception: whatever the study raises when it cannot provide one, for example a
                history with too few completed trials or a problem for which no surrogate
                suits an outcome. The error says why, and an analysis does not catch it.
        """
        ...
