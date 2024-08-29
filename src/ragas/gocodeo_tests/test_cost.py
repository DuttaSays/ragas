import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def setup_mocks():
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        with patch('ragas.utils.get_from_dict') as mock_get_from_dict:
            mock_get_from_dict.side_effect = lambda d, k, default: d.get(k, default)

            with patch('langchain_core.callbacks.base.BaseCallbackHandler') as mock_base_callback_handler:
                mock_base_callback_handler_instance = MagicMock()
                mock_base_callback_handler.return_value = mock_base_callback_handler_instance

                with patch('langchain_core.outputs.LLMResult') as mock_llm_result:
                    mock_llm_result_instance = MagicMock()
                    mock_llm_result.return_value = mock_llm_result_instance

                    with patch('langchain_core.outputs.ChatResult') as mock_chat_result:
                        mock_chat_result_instance = MagicMock()
                        mock_chat_result.return_value = mock_chat_result_instance

                        with patch('langchain_core.outputs.ChatGeneration') as mock_chat_generation:
                            mock_chat_generation_instance = MagicMock()
                            mock_chat_generation.return_value = mock_chat_generation_instance

                            with patch('langchain_core.pydantic_v1.BaseModel') as mock_base_model:
                                mock_base_model_instance = MagicMock()
                                mock_base_model.return_value = mock_base_model_instance

                                yield {
                                    'mock_logger': mock_logger,
                                    'mock_get_from_dict': mock_get_from_dict,
                                    'mock_base_callback_handler': mock_base_callback_handler_instance,
                                    'mock_llm_result': mock_llm_result_instance,
                                    'mock_chat_result': mock_chat_result_instance,
                                    'mock_chat_generation': mock_chat_generation_instance,
                                    'mock_base_model': mock_base_model_instance,
                                }

# happy_path - test_cost_calculation - Test that calculating cost with specified costs works correctly
def test_cost_calculation():
    token_usage = TokenUsage(input_tokens=10, output_tokens=20)
    result = token_usage.cost(cost_per_input_token=0.01, cost_per_output_token=0.02)
    assert result == 0.5

# happy_path - test_equality_identical_objects - Test that equality check works for identical TokenUsage objects
def test_equality_identical_objects():
    token_usage1 = TokenUsage(input_tokens=10, output_tokens=20, model="model_a")
    token_usage2 = TokenUsage(input_tokens=10, output_tokens=20, model="model_a")
    assert token_usage1 == token_usage2

# happy_path - test_is_same_model_identical - Test that is_same_model returns true for identical models
def test_is_same_model_identical():
    token_usage1 = TokenUsage(model="model_a")
    token_usage2 = TokenUsage(model="model_a")
    assert token_usage1.is_same_model(token_usage2)

# happy_path - test_openai_token_usage_parsing - Test that token usage is parsed correctly for OpenAI model
def test_openai_token_usage_parsing(setup_mocks):
    llm_result = setup_mocks['mock_llm_result']
    llm_result.llm_output = {'token_usage.completion_tokens': 50, 'token_usage.prompt_tokens': 30}
    result = get_token_usage_for_openai(llm_result)
    assert result.input_tokens == 30
    assert result.output_tokens == 50

# happy_path - test_on_llm_end_aggregation - Test that token usage is aggregated correctly in on_llm_end
def test_on_llm_end_aggregation(setup_mocks):
    handler = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    response = setup_mocks['mock_llm_result']
    response.llm_output = {'token_usage.completion_tokens': 20, 'token_usage.prompt_tokens': 10}
    handler.on_llm_end(response)
    assert handler.usage_data == [TokenUsage(input_tokens=10, output_tokens=20)]

# edge_case - test_add_different_model - Test that adding TokenUsage objects with different models raises ValueError
def test_add_different_model():
    token_usage1 = TokenUsage(input_tokens=10, output_tokens=20, model="model_a")
    token_usage2 = TokenUsage(input_tokens=5, output_tokens=15, model="model_b")
    with pytest.raises(ValueError, match="Cannot add TokenUsage objects with different models"):
        token_usage1 + token_usage2

# edge_case - test_total_cost_no_costs_provided - Test that total_cost raises ValueError when no costs are provided
def test_total_cost_no_costs_provided():
    handler = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    handler.usage_data = [TokenUsage(input_tokens=10, output_tokens=20, model="model_a")]
    with pytest.raises(ValueError, match="No cost table or cost per token provided"):
        handler.total_cost()

# edge_case - test_total_tokens_multiple_models - Test that total_tokens returns correct sum when multiple models are used
def test_total_tokens_multiple_models():
    handler = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    handler.usage_data = [
        TokenUsage(input_tokens=10, output_tokens=20, model="model_a"),
        TokenUsage(input_tokens=5, output_tokens=15, model="model_b")
    ]
    result = handler.total_tokens()
    assert result == [
        TokenUsage(input_tokens=10, output_tokens=20),
        TokenUsage(input_tokens=5, output_tokens=15)
    ]

# edge_case - test_anthropic_no_metadata - Test that get_token_usage_for_anthropic returns zero tokens when no response metadata is present
def test_anthropic_no_metadata(setup_mocks):
    llm_result = setup_mocks['mock_llm_result']
    llm_result.generations = [[]]
    result = get_token_usage_for_anthropic(llm_result)
    assert result.input_tokens == 0
    assert result.output_tokens == 0

# edge_case - test_total_cost_with_per_model_costs - Test that total_cost calculates correctly with per model costs
def test_total_cost_with_per_model_costs():
    handler = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    handler.usage_data = [TokenUsage(input_tokens=10, output_tokens=20, model="model_a")]
    per_model_costs = {"model_a": (0.01, 0.02)}
    result = handler.total_cost(per_model_costs=per_model_costs)
    assert result == 0.5

# edge_case - test_total_tokens_list_return - Test that total_tokens returns a list when multiple models are used
def test_total_tokens_list_return():
    handler = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    handler.usage_data = [
        TokenUsage(input_tokens=10, output_tokens=20, model="model_a"),
        TokenUsage(input_tokens=5, output_tokens=15, model="model_b")
    ]
    result = handler.total_tokens()
    assert isinstance(result, list)
    assert result == [
        TokenUsage(input_tokens=10, output_tokens=20),
        TokenUsage(input_tokens=5, output_tokens=15)
    ]

