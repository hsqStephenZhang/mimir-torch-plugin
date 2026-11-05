# MimIR torch plugin

This repository contains the `torch` plugin for MimIR. It exposes PyTorch
operator-shaped axioms and decomposes them into MimIR tensor operations.

Depends on https://github.com/hsqStephenZhang/mimir-runtime-plugin

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

## Layout

- `torch.mim` declares the public `torch.*` axioms, their `*_impl` lambdas and the
  phases; every annex lives here. It requires the `mod ... end` module syntax of
  MimIR master (commit `ebd86c79` or newer).
- `helper.mim` holds the scalar dictionaries (`Arithmetic`, `Floating`,
  `Reduction`), shape arithmetic, index maps and element functions. `torch.mim`
  splices it with `pub import "helper.mim" as *`, so the helpers stay reachable
  as `torch.<name>` for tests and other plugins. The import is resolved relative
  to `torch.mim`, which is why `CMakeLists.txt` stages and installs `helper.mim`
  next to it. A helper that calls one of the `*_impl` modules stays in
  `torch.mim`.
- The recurrent operators (`rnn`, `gru`, `lstm`) build on `tensor.scan`, which
  upstream MimIR master does not provide yet; they need a `tensor` plugin with
  the scan support of the `feat/plugin-support-v2` branch.
