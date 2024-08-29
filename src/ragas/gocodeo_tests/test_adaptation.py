import pytest
from unittest.mock import MagicMock, patch
from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics.base import MetricWithLLM
from langchain_core.language_models import BaseLanguageModel

@pytest.fixture
def mock_llm_factory():
    with patch('ragas.llms.llm_factory', return_value=MagicMock(spec=LangchainLLMWrapper)) as mock:
        yield mock

@pytest.fixture
def mock_langchain_llm_wrapper():
    with patch('ragas.llms.base.LangchainLLMWrapper', return_value=MagicMock(spec=LangchainLLMWrapper)) as mock:
        yield mock

@pytest.fixture
def mock_base_ragas_llm():
    return MagicMock(spec=BaseRagasLLM)

@pytest.fixture
def mock_base_language_model():
    return MagicMock(spec=BaseLanguageModel)

@pytest.fixture
def mock_metric_with_llm():
    return MagicMock(spec=MetricWithLLM)

@pytest.fixture
def setup_mocks(mock_llm_factory, mock_langchain_llm_wrapper, mock_base_ragas_llm, mock_base_language_model, mock_metric_with_llm):
    return {
        'llm_factory': mock_llm_factory,
        'LangchainLLMWrapper': mock_langchain_llm_wrapper,
        'BaseRagasLLM': mock_base_ragas_llm,
        'BaseLanguageModel': mock_base_language_model,
        'MetricWithLLM': mock_metric_with_llm,
    }

# happy_path - test_adapt_with_llm - Test that the function adapts metrics using a provided BaseRagasLLM instance.
def test_adapt_with_llm(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    language = 'en'
    llm = setup_mocks['BaseRagasLLM']
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics[0].llm == setup_mocks['LangchainLLMWrapper']

# happy_path - test_adapt_without_llm - Test that the function adapts metrics using a newly created LLM when none is provided.
def test_adapt_without_llm(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    language = 'en'
    llm = None
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics[0].llm == setup_mocks['LangchainLLMWrapper']

# happy_path - test_adapt_calls_metric_adapt - Test that the function correctly calls the adapt method on metrics that have it.
def test_adapt_calls_metric_adapt(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    metrics[0].adapt = MagicMock()
    language = 'en'
    llm = None
    cache_dir = 'cache_dir'

    adapt(metrics, language, llm, cache_dir)

    metrics[0].adapt.assert_called_once_with(language, cache_dir=cache_dir)

# happy_path - test_adapt_saves_metrics - Test that the function saves metrics after adaptation if they have a save method.
def test_adapt_saves_metrics(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    metrics[0].save = MagicMock()
    language = 'en'
    llm = None
    cache_dir = 'cache_dir'

    adapt(metrics, language, llm, cache_dir)

    metrics[0].save.assert_called_once_with(cache_dir=cache_dir)

# happy_path - test_adapt_with_existing_llm - Test that the function handles metrics with existing llm correctly when a new llm is provided.
def test_adapt_with_existing_llm(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    metrics[0].llm = 'existing LLM'
    language = 'en'
    llm = setup_mocks['BaseRagasLLM']
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics[0].llm == setup_mocks['LangchainLLMWrapper']

# edge_case - test_adapt_invalid_llm_type - Test that the function raises a ValueError if llm is neither None nor a BaseLanguageModel.
def test_adapt_invalid_llm_type(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    language = 'en'
    llm = 'InvalidType'
    cache_dir = None

    with pytest.raises(ValueError, match='llm must be either None or a BaseLanguageModel'):
        adapt(metrics, language, llm, cache_dir)

# edge_case - test_adapt_empty_metrics - Test that the function handles an empty metrics list without errors.
def test_adapt_empty_metrics(setup_mocks):
    metrics = []
    language = 'en'
    llm = None
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics == []

# edge_case - test_adapt_metrics_without_adapt_method - Test that the function handles metrics that do not have an adapt method gracefully.
def test_adapt_metrics_without_adapt_method(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    del metrics[0].adapt
    language = 'en'
    llm = None
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics[0].llm == setup_mocks['LangchainLLMWrapper']

# edge_case - test_adapt_no_change_with_existing_llm - Test that the function does not alter metrics with a non-None llm when no new llm is provided.
def test_adapt_no_change_with_existing_llm(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    metrics[0].llm = 'existing LLM'
    language = 'en'
    llm = None
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics[0].llm == 'existing LLM'

# edge_case - test_adapt_with_none_cache_dir - Test that the function adapts metrics correctly even when cache_dir is None.
def test_adapt_with_none_cache_dir(setup_mocks):
    metrics = [setup_mocks['MetricWithLLM']]
    language = 'en'
    llm = None
    cache_dir = None

    adapt(metrics, language, llm, cache_dir)

    assert metrics[0].llm == setup_mocks['LangchainLLMWrapper']

