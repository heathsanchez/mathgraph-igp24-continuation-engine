"""Backward-compatible wrapper for the v102 artifact builder."""
from build_colab_artifact import build

if __name__ == "__main__":
    for path in build(): print(path)

