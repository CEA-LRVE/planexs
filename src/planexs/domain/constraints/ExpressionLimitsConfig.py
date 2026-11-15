"""The limits on the size of a constraint expression, as a configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExpressionLimitsConfig(BaseModel):
    """The two limits that `validateExpression` applies to an expression (specification 3.4,
    10.8).

    A constraint is an arithmetic condition on a handful of parameters, not a program, and an
    expression that is hostile or only mistaken (a megabyte of `x + x + ...`, ten thousand
    nested minus signs) could exhaust the parser. These two settings say how large an
    expression may be before it is refused: how long its text is, and how deeply its operations
    nest. They are the only policy values of the validator: everything else it decides is the
    grammar, which is not a setting.

    They are the configuration of the validator, not of a registered component: this class has
    no `type` field, is not registered in any kind and is not a `ComponentConfig`, like
    `RuntimeConfig` and `EncoderConfig`. A `NonlinearConstraint` holds it as its `limits`
    field, so the limits that a constraint was validated with travel with it into every saved
    study, and `validateConfig` reads it like any other model, with the same field paths in
    its errors (for example `limits.max_depth`).

    The defaults are the values that the validator used before they became settings, so that
    a constraint that sets nothing is validated exactly as it was. They are written in the
    fields below and nowhere else.

    A configuration is frozen: it is shared by the validator and both evaluators and must not
    change under them. Build a new one to change a setting.

    Attributes:
        max_length: The largest number of characters of an expression.
        max_depth: The deepest level at which an operation, a call or a name may be nested.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_length: int = Field(
        default=1000,
        ge=1,
        description=(
            "The largest number of characters that an expression may have, in characters of "
            "the text with the whitespace around it removed. A longer expression is refused "
            "before it is parsed, so the limit is also what bounds the work of the parser. "
            "The default is 1000: about a hundred terms of a sum, far beyond any condition "
            "written by hand, and small enough that a hostile or accidental expression "
            "cannot exhaust the parser. Raise it for a constraint that is generated, such as "
            "a sum over hundreds of parameters."
        ),
    )

    max_depth: int = Field(
        default=200,
        ge=0,
        description=(
            "The deepest level at which an operation, a call or a name may be nested in an "
            "expression, in levels of its syntax tree: the whole expression is level 0, its "
            "operands are level 1, theirs level 2, and so on, so 0 allows a bare parameter "
            "name and nothing else. Parentheses that group nothing are not a level. An "
            "expression nested more deeply is refused. The validator and both evaluators "
            "walk the tree recursively, one Python stack frame per level, so this is also "
            "what keeps them far from the interpreter's recursion limit (1000 by default, "
            "less the frames already in use). The default is 200: a fifth of that limit, "
            "and far deeper than any expression written by hand. Keep a raised value well "
            "below the recursion limit: the compiled form of a constraint is validated once, "
            "when it is built, and walked at every evaluation from deeper in the call stack "
            "(inside the optimizer), so an expression close to the limit can be accepted and "
            "still overflow the stack there."
        ),
    )
