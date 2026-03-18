import tiktoken

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Counts the number of tokens in a string.
    Using tiktoken as a proxy for token length if Gemini doesn't provide it directly.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to a simple approximation if tiktoken fails
        return len(text.split()) * 1.3
