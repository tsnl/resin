import resin.gpu.kernel


def test_shader_template_consts():
    loaded_shader = resin.gpu.kernel.Test1Kernel(test_constant=42)

    assert "const TEST_CONSTANT = u32(42);" in loaded_shader.wgsl()
    print(loaded_shader)
