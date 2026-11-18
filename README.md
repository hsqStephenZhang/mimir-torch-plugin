# MimIR torch plugin

The `torch` plugin for MimIR: PyTorch operator-shaped axioms that decompose into MimIR tensor operations.
Requires the [runtime plugin](https://github.com/hsqStephenZhang/mimir-runtime-plugin), which several checked
operators use.

## Build

Checked out under MimIR's `extra/`, the plugin is built with MimIR. Standalone, against an installed MimIR:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/mimir/install
cmake --build build -j14
```

## Layout

- `torch.mim`: the public `torch.*` axioms, their `*_impl` lambdas, the phases and every annex. Needs the
  `mod ... end` module syntax of MimIR master (`ebd86c79` or newer).
- `helper.mim`: scalar dictionaries (`Arithmetic`, `Floating`, `Reduction`), shape arithmetic, index maps and
  element functions. `torch.mim` imports it with `pub import "helper.mim" as *`, so the helpers are reachable as
  `torch.<name>`; the import is relative to `torch.mim`, hence `CMakeLists.txt` stages `helper.mim` next to it.
- The recurrent operators (`rnn`, `gru`, `lstm`) need `tensor.scan`, which upstream MimIR master does not provide
  yet (see the `feat/plugin-support-v2` branch).

## Nominal tensors

Tensor arguments and results are `tensor.Tensor rank shape element_type`, keeping logical rank including rank zero
(singleton axes do not collapse). Shapes, coordinates, affine maps and scalar pairs remain arrays. Use
`tensor.scalar` / `tensor.item` at scalar boundaries and `tensor.from_array`, `tensor.splat` or `tensor.generate` to
build tensors; external buffers enter as Tensor-typed parameters.

- `torch.creation.scalar_tensor` returns `tensor.Tensor 0 () T`.
- `torch.reduction.sum`, `sum_typed`, `mean`, `logsumexp`, `amax`, `vector_norm` and `var_mean`'s two tensor fields
  return a dependent `ReductionTensor` that keeps the `keepdim` shape; scalar-returning schemas such as `sum_all`
  stay scalar.
- Unreduced losses are tensors; reduced losses are scalars.
