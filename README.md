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

## Nominal tensors

Tensor arguments and results use `tensor.Tensor rank shape element_type`.
Shape arrays, coordinate vectors, affine maps and scalar pairs remain arrays.
Singleton axes and rank-zero tensors no longer collapse: callers must use
`tensor.scalar` / `tensor.item` at scalar boundaries and `tensor.from_array`,
`tensor.splat` or `tensor.generate` to construct tensor values. External buffers
enter as Tensor-typed parameters, not through the removed `tensor.buf`.

The one-shot mechanical conversion is in `scripts/port_nominal.py`; run its
passes with `uv run --no-sync python`. It infers ranks from shape declarations,
preserves non-tensor arrays, updates bitwise and option spellings, and converts
common access, splat and dependent-keepdim patterns. Constructors, indexing,
scan state types and test literals were reviewed separately. It is not an
idempotent formatter and must not be rerun on already converted files.

Public axiom names, explicit argument groups and argument order are unchanged.
Their tensor types now preserve logical rank, including rank zero.
`torch.creation.scalar_tensor` returns `tensor.Tensor 0 () T` instead of `T`.
`torch.reduction.sum`, `sum_typed`, `mean`, `logsumexp`, `amax`,
`vector_norm` and the two tensor fields of `var_mean` use a dependent
`ReductionTensor` result that preserves the selected `keepdim` shape. Existing
explicit scalar-returning schemas such as `sum_all` remain scalar-returning.
Unreduced losses remain tensors; reduced loss schemas retain scalar results.

### Upstream blockers

Three tests retain their original RUN/CHECK assertions but are marked XFAIL:

- `generic_cross_entropy.mim`: core `World::match` dispatch calls `inj` on a
  payload already having a nominal Tensor type. `src/mim/world.cpp:1090`
  reaches the nominal-construction branch at line 986 instead of preserving
  the value. Run `build/bin/mim extra/torch/scripts/nominal_union.mim -o /dev/null`
  from the MimIR root; this reproducer does not import the torch plugin.
- `cross_entropy_invalid_smoothing.mim`: the same dispatch error occurs before
  the intended invalid-smoothing diagnostic.
- `nominal_exec.mim`: combining `dynamic_relu1` and `dynamic_sum1` crashes in
  `SEO::Analysis::non_mem_free_vars` (`src/mim/plug/mem/phase/seo.cpp:252`).
  Its `isa_mem` call reaches `Axm::get` with a null type via
  `include/mim/plug/core/core.h:127`. Run
  `build/bin/mim extra/torch/scripts/nominal_seo.mim -p opt -p ll` from the
  MimIR root. Each of the eight execution-test functions compiles in isolation;
  linking those individual LLVM modules passes the original numerical harness.

`generic_reductions.mim` passes, including singleton keepdim axes and mixed
dtype reductions. No MimIR core or runtime-plugin files were changed.
