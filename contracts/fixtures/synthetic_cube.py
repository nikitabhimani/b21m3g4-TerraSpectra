"""CLI: python contracts/fixtures/synthetic_cube.py out.tif [--size 256]"""

import argparse

from terraspectra_contracts.fixtures import write_synthetic_cube

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(write_synthetic_cube(a.out, a.size, a.size, a.seed))
