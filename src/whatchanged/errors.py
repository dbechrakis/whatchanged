"""Errors raised by whatchanged. Each message names the problem."""


class WhatChangedError(ValueError):
    """The input cannot be analysed. The message says what is wrong and where."""
