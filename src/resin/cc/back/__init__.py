from .. import ir


def build(modules: dict[str, ir.Module]):
    for module in modules.values():
        print(f"module {module.name}:")
        for func in module.functions.values():
            print(f"\tfn {func.raw.__name__} :: {func.signature}")
        for struct in module.structs.values():
            print(f"\tstruct {struct.raw.__name__}")

    print("TODO: implement resin.cc.back.build()")
