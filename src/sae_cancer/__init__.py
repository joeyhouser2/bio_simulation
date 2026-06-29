"""ESMC-SAE cancer mutation interpretability pipeline.

See initial_writeup.pdf for the project brief and CLAUDE.md for conventions.
"""

# Windows DLL guard: importing pyarrow AFTER torch causes a native access violation, and
# the esm package imports pandas (-> pyarrow) lazily inside esm.tokenization. Importing
# pyarrow here — before any submodule pulls torch — guarantees the safe order for every
# `import sae_cancer.*` entry point.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
