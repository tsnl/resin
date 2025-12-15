from .basic import round_up_to_po2, camel_to_snake


def test_round_up_to_po2() -> None:
    assert round_up_to_po2(0) == 1
    assert round_up_to_po2(1) == 1
    assert round_up_to_po2(2) == 2
    assert round_up_to_po2(3) == 4
    assert round_up_to_po2(4) == 4
    assert round_up_to_po2(5) == 8


def test_camel_to_snake() -> None:
    assert camel_to_snake("CamelCase") == "camel_case"
    assert camel_to_snake("HttpRequest") == "http_request"
    assert camel_to_snake("simpleTest") == "simple_test"
    assert camel_to_snake("Already_Snake_Case") == "already_snake_case"
    assert camel_to_snake("Almost_PascalCase") == "almost_pascal_case"
