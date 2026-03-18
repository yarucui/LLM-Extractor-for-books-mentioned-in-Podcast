def count_words(text: str) -> int:
    """
    Counts the number of words in a string.
    """
    if not text:
        return 0
    return len(text.split())
