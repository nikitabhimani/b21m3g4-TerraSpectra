"""CLI: python contracts/fixtures/stub_model.py stub_model.pt"""

import sys

from terraspectra_contracts.fixtures import export_stub_model

if __name__ == "__main__":
    print(export_stub_model(sys.argv[1] if len(sys.argv) > 1 else "stub_model.pt"))
