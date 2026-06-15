import resin.gpu.shaders


def test_shader_template_consts():
    loaded_shader = resin.gpu.shaders.load_shader(
        "test-shader",
        {"TEST_CONSTANT": "42"},
    )

    assert "const TEST_CONSTANT: u32 = 42;" in loaded_shader
    print(loaded_shader)
