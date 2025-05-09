from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionRequest as _ChatCompletionRequest,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionResponse as _ChatCompletionResponse,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionStreamResponse as _ChatCompletionStreamResponse,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    CompletionRequest as _CompletionRequest,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    CompletionResponse as _CompletionResponse,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    CompletionStreamResponse as _CompletionStreamResponse,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    ErrorResponse as _ErrorResponse,
)
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class ChatCompletionRequest(_ChatCompletionRequest):
    """ChatCompletionRequest is the request body for the chat completion API.

    This model is compatible with vLLM's OpenAI API models.
    """

    pass


@PublicAPI(stability="alpha")
class CompletionRequest(_CompletionRequest):
    """CompletionRequest is the request body for the completion API.

    This model is compatible with vLLM's OpenAI API models.
    """

    pass


@PublicAPI(stability="alpha")
class ChatCompletionStreamResponse(_ChatCompletionStreamResponse):
    """ChatCompletionStreamResponse is the response body for the chat completion API.

    This model is compatible with vLLM's OpenAI API models.
    """

    pass


@PublicAPI(stability="alpha")
class ChatCompletionResponse(_ChatCompletionResponse):
    """ChatCompletionResponse is the response body for the chat completion API.

    This model is compatible with vLLM's OpenAI API models.
    """

    pass


@PublicAPI(stability="alpha")
class CompletionStreamResponse(_CompletionStreamResponse):
    """CompletionStreamResponse is the response body for the completion API.

    This model is compatible with vLLM's OpenAI API models.
    """

    pass


@PublicAPI(stability="alpha")
class CompletionResponse(_CompletionResponse):
    """CompletionResponse is the response body for the completion API.

    This model is compatible with vLLM's OpenAI API models.
    """

    pass


@PublicAPI(stability="alpha")
class ErrorResponse(_ErrorResponse):
    """The returned response in case of an error."""

    pass
