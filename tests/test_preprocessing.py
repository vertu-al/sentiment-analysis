from src.preprocessing import preprocess_text


def test_lowercase():
    text = "HELLO WORLD"
    result = preprocess_text(text)

    assert result == "hello world"


def test_remove_punctuation():
    text = "Hello!!!"

    result = preprocess_text(text)

    assert result == "hello"


def test_keep_negation():
    text = "This is not good."

    result = preprocess_text(text)

    assert "not" in result


def test_empty_string():
    result = preprocess_text("")

    assert result == ""


def test_non_string():
    import pytest

    with pytest.raises(TypeError):
        preprocess_text(42)