import pytest

from llmproc.process_forking import ProcessForkingMixin
from tests.conftest import create_test_llmprocess_directly


class DummyProcess(ProcessForkingMixin):
    pass


@pytest.mark.asyncio
async def test_create_snapshot():
    process = create_test_llmprocess_directly()
    process.state.append({"role": "user", "content": "hi"})
    snapshot = process._create_snapshot()
    assert snapshot.state == process.state
    assert snapshot.state is not process.state
