from .ir import IrProgram


def optimize(program: IrProgram) -> IrProgram:
    # TODO: kernel fusion, dead-buffer elimination, etc.
    return program
