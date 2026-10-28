"""Explicit serving-tokenizer protocol fixtures for synthetic task LMs."""

import pytest

from daemon.graph.inference.test_reflection import provider as synonym_provider
from yaw.daemon.graph.inference import ServingTokenCounter

__all__ = ["synonym_provider"]


class ProtocolTokenCounter:
    """Count through a real loopback tokenizer, independent of the synthetic task LM."""

    def __init__(self, url):
        self.serving = ServingTokenCounter(url, "reflection-model")
        self.identity = self.serving.identity
        self.context_tokens = self.serving.context_tokens

    def __call__(self, messages, **kwargs):
        return self.serving(messages, **kwargs)


@pytest.fixture
def synonym_token_counter(synonym_provider):
    return ProtocolTokenCounter(synonym_provider[0])
