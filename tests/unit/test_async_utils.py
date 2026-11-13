# tests/unit/test_async_utils.py
"""Юнит-тесты для декоратора @retry."""

import pytest

from backend.utils.async_utils import retry


class TestRetry:
    async def test_success_on_first_attempt(self):
        calls = []

        @retry(max_attempts=3, delay=0.0)
        async def ok():
            calls.append(1)
            return "done"

        assert await ok() == "done"
        assert len(calls) == 1

    async def test_retries_then_succeeds(self):
        calls = []

        @retry(max_attempts=3, delay=0.0)
        async def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")
            return "done"

        assert await flaky() == "done"
        assert len(calls) == 3

    async def test_gives_up_after_max_attempts(self):
        calls = []

        @retry(max_attempts=2, delay=0.0)
        async def always_fails():
            calls.append(1)
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            await always_fails()
        assert len(calls) == 2

    async def test_delay_grows_with_backoff(self):
        import backend.utils.async_utils as au

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        original = au.asyncio.sleep
        au.asyncio.sleep = fake_sleep
        try:

            @retry(max_attempts=3, delay=1.0, backoff=2.0)
            async def always_fails():
                raise RuntimeError("x")

            with pytest.raises(RuntimeError):
                await always_fails()
        finally:
            au.asyncio.sleep = original

        # 1с → 2с (третья попытка — последняя, сна после неё нет)
        assert sleeps == [1.0, 2.0]
