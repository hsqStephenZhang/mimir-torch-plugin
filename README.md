# MimIR torch plugin

This repository contains the `torch` plugin for MimIR. It exposes PyTorch
operator-shaped axioms and decomposes them into MimIR tensor operations.

## Standalone build

Install MimIR so that its CMake package is discoverable, then configure this
repository:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/mimir/install
cmake --build build -j14
```

When the repository is checked out under MimIR's `extra/` directory, MimIR
discovers and builds it automatically. The runtime plugin must be available
alongside this plugin because several checked operators use runtime checks.
