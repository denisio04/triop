"""Dump full failure reports for the pilot test to /tmp for post-mortem."""
import os

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        os.makedirs("/tmp/opencode", exist_ok=True)
        with open("/tmp/opencode/triop-test-fail.log", "a") as f:
            f.write(f"\n=== {item.nodeid} ===\n")
            f.write(str(call.excinfo.getrepr(style="long")) + "\n")
