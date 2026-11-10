# SPDX-License-Identifier: Apache-2.0
"""Worker failures reach the caller with a usable reason."""

from sglang.multimodal_gen.runtime.managers.gpu_worker import describe_error


def _raise(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as e:
        return e


def test_message_is_kept() -> None:
    assert describe_error(_raise(RuntimeError("out of memory"))) == "out of memory"


def test_bare_assert_names_type_and_site() -> None:
    def capture():
        # A bare `assert` raises this (pytest rewrites asserts in tests to
        # carry a message, so raise it directly).
        raise AssertionError()

    try:
        capture()
    except AssertionError as e:
        text = describe_error(e)
    assert text.startswith("AssertionError at test_worker_error_messages.py:")
