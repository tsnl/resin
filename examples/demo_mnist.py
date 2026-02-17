from dataclasses import dataclass

import resin.graph as rg
import resin.optim as ro


@dataclass
class BabyMlp:
    l1: ro.Linear
    l2: ro.Linear
    l3: ro.Linear

    def __call__(self, x: rg.Node) -> rg.Node:
        w1 = ro.Parameter((784, 128), dtype="fp32")
        w2 = ro.Parameter((128, 10), dtype="fp32")
        return (x @ w1).relu() @ w2


def main():
    pass


if __name__ == "__main__":
    main()
