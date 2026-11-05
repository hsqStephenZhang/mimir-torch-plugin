"""One-shot migration of the pre-nominal dialect's tensor annotations.

Run with uv run --no-sync python scripts/port_nominal.py PATH...
Shape, coordinate, affine-map and scalar-pair arrays are deliberately retained.
Constructor and element-access changes require a semantic review after this pass.
"""

import re
import subprocess
import sys
from pathlib import Path


def split_top(text, delimiter=","):
    parts = []
    start = depth = 0
    for i, char in enumerate(text):
        if char in "([{«‹":
            depth += 1
        elif char in ")]}»›":
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def atom(text):
    if re.fullmatch(r"[\w.#]+|\([^;]*\)", text, re.S):
        return text
    return f"({text})"


def shape_rank(shape, env):
    shape = shape.strip()
    if shape.startswith("torch.torch_"):
        return shape_rank(shape[len("torch."):], env)
    if shape == "()":
        return "0"
    if shape.startswith("(") and shape.endswith(")") and "," in shape:
        return str(len([p for p in split_top(shape[1:-1]) if p]))
    pack = re.fullmatch(r"‹\w+:\s*([^;]+);.*›", shape, re.S)
    if pack:
        return pack[1]
    parts = split_top(shape)
    if len(parts) > 1:
        return str(len([p for p in parts if p]))
    if shape in env:
        return env[shape]
    if shape.startswith("shapes#"):
        return env.get("shapes", "r")
    if shape.startswith("s#") or shape.startswith("torch_numel "):
        return "1"
    if shape.startswith("vec.cat "):
        args = split_top(shape[len("vec.cat "):].strip()[1:-1])
        return f"{shape_rank(args[0], env)} + {shape_rank(args[1][1:-1], env) if args[1].startswith('(') else shape_rank(args[1], env)}"
    if shape.startswith("torch_multi_reduce_shape"):
        args = split_top(shape[shape.index("(") + 1:-1])
        return f"torch_reduced_rank {atom(args[1])}"
    if shape.startswith("torch_matmul_output_shape"):
        return "torch_matmul_output_rank (rl, rr)"
    if shape.startswith("torch_reduce_shape"):
        args = split_top(shape[shape.index("(") + 1:-1])
        return f"{shape_rank(args[0], env)} - 1"
    if shape.startswith("torch_unfold_shape"):
        return "r + 1"
    if shape.startswith("torch_concat_shape"):
        return env.get("s_singleton", "r")
    fixed = {
        "tensor.conv_shape": "4",
        "conv3d_shape": "5",
        "conv_transpose2d_shape": "4",
        "conv_transpose3d_shape": "5",
        "torch_conv1d_shape": "3",
        "torch_conv_transpose1d_shape": "3",
        "torch_pool1d_shape": "3",
    }
    for fn, rank in fixed.items():
        if shape.startswith(fn):
            return rank
    if re.match(r"torch_(?:multi_keepdim|keepdim|narrow|pad|permute|repeat|transpose|adaptive_pool[123]d)_shape\b", shape):
        return env.get("s", "r")
    if re.fullmatch(r"\d+|\w+|\w+ \* \w+", shape):
        return "1"
    raise ValueError(f"unknown shape rank: {shape!r}")


def convert_block(block, helper, test):
    env = {}
    for match in re.finditer(r"(\w+(?:\s+\w+)*)\s*:\s*«([^«»;]+);\s*Nat»", block):
        for name in match[1].split():
            env[name] = match[2].strip()
    nested = re.search(r"shapes:\s*«ni;\s*«([^;]+); Nat»»", block)
    if nested:
        env["shapes"] = nested[1].strip()
    for name, rank in {"output_shape": "rb + 1", "out_shape": "out_rank", "loss_shape": "r - 1"}.items():
        env.setdefault(name, rank)
    if "torch_group_norm_suffix" in block:
        env["normalized_shape"] = "r - 1"
    # Local shape aliases whose annotation is implicit in the old source.
    for match in re.finditer(r"let (\w+) = (vec.cat \(batch_shape, \([^;]+\)\)|torch_reduce_shape [^;]+);", block):
        env[match[1]] = shape_rank(match[2], env)
    stack = []
    edits = []
    for i, char in enumerate(block):
        if char == "«":
            stack.append(i)
        elif char == "»":
            j = stack.pop()
            body = block[j + 1:i]
            parts = split_top(body, ";")
            if len(parts) != 2:
                continue
            shape, elem = parts
            if not re.fullmatch(r"(?:F#A#R#T|A#R#T|R#T|T|Ti|Tin|D#Input|D#Output|Bool|I64|math.F32|math.F64|F32|F64|I32|Nat|Idx bound)", elem):
                continue
            if elem == "Nat" and not test:
                continue
            if elem == "I64" and (shape in ("nr", "r", "r + 2", "r + 3") or (helper and shape in ("1", "2", "4", "5"))):
                continue
            if helper and shape == "2":
                continue
            if test and elem == "Nat" and (
                re.search(r"\b(?:s|s_in|s_out|shape|dims|stride|padding|dilation|kernel|actual|expected|repeats|dimensions|output_size|batch_shape|normalized_shape)\s*:\s*$", block[:j])
                or "Arithmetic =" in block
            ):
                continue
            rank = shape_rank(shape, env)
            edits.append((j, i + 1, f"(tensor.Tensor {atom(rank)} {atom(shape)} {atom(elem)})"))
    for start, end, replacement in reversed(edits):
        block = block[:start] + replacement + block[end:]
    return block


def convert(path, revision=None):
    text = subprocess.check_output(
        ["git", "-C", str(path.parent), "show",
         f"{revision}:{'lit/' if path.parent.name == 'lit' else ''}{path.name}"], text=True
    ) if revision else path.read_text()
    text = re.sub(r"core.bit2.(and|or|xor)_\b", r"core.bit2.\1", text)
    text = re.sub(r"tensor.shape [45] (\w+)", r"tensor.shape \1", text)
    chunks = re.split(r"(?=^(?:(?:pub|anx|extern|priv)\s+)*(?:lam|fun|con|axm|let)\s)", text, flags=re.M)
    text = "".join(convert_block(b, path.name == "helper.mim", path.parent.name == "lit") for b in chunks)
    text = re.sub(r"option.Opt\s+(?!@)", "option.Opt@", text)
    while match := re.search(r"option.none\s+@", text):
        start = end = match.end()
        if text[start] == "(":
            depth = 1
            end += 1
            while depth:
                depth += (text[end] == "(") - (text[end] == ")")
                end += 1
        else:
            end += re.match(r"[\w.]+", text[start:]).end()
        text = text[:match.start()] + f"(option.Opt@{text[start:end]})#None" + text[end:]
    text = text.replace("option.some", "option.Opt#Some")
    text = re.sub(r"\| (i|bound|actual_weight): [^\n]+? =>", r"| Some \1 =>", text)
    text = text.replace("| _: [] =>", "| None =>")
    if path.parent.name == "lit" and "tensor.Tensor" in text and "plugin tensor;" not in text:
        text = text.replace("plugin torch;", "plugin tensor;\nplugin torch;")
    path.write_text(text)


def boundaries(path):
    text = path.read_text()
    text = text.replace("input#coords", "tensor.get (input, coords)")
    text = text.replace("flat#(i,)", "tensor.get (flat, i)")
    text = text.replace("reduced#(0_2,)", "tensor.get (reduced, 0_2)")

    def splat(match):
        size, value = match.groups()
        elem = "F#A#R#T" if "F#" in value else "A#R#T"
        return f"(tensor.splat @({elem}, 1) ({size}, {value}))"

    text = re.sub(r"‹(n|cout|channels|actual_channels|classes|examples); ([^›]+)›", splat, text)
    path.write_text(text)


def keepdim(path):
    chunks = re.split(r"(?=^anx lam )", path.read_text(), flags=re.M)
    for i, block in enumerate(chunks):
        if "(reduced, kept)#keepdim" not in block and "torch_apply_keepdim" not in block:
            continue
        elem = "D#Output" if "(D: Reduction)" in block else (
            "A#R#T" if "(A: Arithmetic)" in block else "F#A#R#T"
        )
        block = block.replace(
            "(reduced, kept)#keepdim",
            f"torch_apply_keepdim @({elem}, r, nr) (s, dims, keepdim) reduced",
        )
        block = re.sub(r" +let kept = tensor.reshape\b[^;]+;\n", "", block)
        chunks[i] = block
    path.write_text("".join(chunks))


def returns(path):
    chunks = re.split(r"(?=^anx lam )", path.read_text(), flags=re.M)
    for i, block in enumerate(chunks):
        if "torch_apply_keepdim" not in block:
            continue
        elem = "D#Output" if "(D: Reduction)" in block else (
            "A#R#T" if "(A: Arithmetic)" in block else "F#A#R#T"
        )
        dims = "torch_canonical_dims @(r, nr) dimensions"
        result = (
            f"tensor.Tensor (torch_reduced_rank ({dims}))\n"
            f"        (torch_multi_reduce_shape (s, {dims})) {elem} ∪\n"
            f"    tensor.Tensor r (torch_multi_keepdim_shape (s, {dims})) {elem}"
        )
        header, _, body = block.partition(" =")
        block = header + f": {result} =" + body
        chunks[i] = re.sub(r"lam (\w+)\(\) =", rf"lam \1(): {result} =", block)
    path.write_text("".join(chunks))


def dependent_keepdim(path):
    text = path.read_text()
    pattern = (
        r"→ \(\(tensor.Tensor \(torch_reduced_rank .*?"
        r"\)\s+(A#R#T|D#Output|F#A#R#T)\)\)#keepdim;"
    )
    text = re.sub(
        pattern,
        lambda m: f"→ ReductionTensor @({m[1]}, r, nr)\n"
                  "        (s, torch_canonical_dims @(r, nr) dimensions, keepdim);",
        text, flags=re.S,
    )
    pattern = (
        r"tensor.Tensor \(torch_reduced_rank \(torch_canonical_dims @\(r, nr\) dimensions\)\)\n"
        r"\s+\(torch_multi_reduce_shape \(s, torch_canonical_dims @\(r, nr\) dimensions\)\) (A#R#T|D#Output|F#A#R#T) ∪\n"
        r"\s+tensor.Tensor r \(torch_multi_keepdim_shape \(s, torch_canonical_dims @\(r, nr\) dimensions\)\) \1"
    )
    text = re.sub(
        pattern,
        lambda m: f"ReductionTensor @({m[1]}, r, nr)\n"
                  "            (s, torch_canonical_dims @(r, nr) dimensions, keepdim)",
        text,
    )
    path.write_text(text.replace("\n : ReductionTensor", "\n  : ReductionTensor"))


if __name__ == "__main__":
    args = sys.argv[1:]
    passes = {
        "--boundaries": boundaries, "--keepdim": keepdim,
        "--returns": returns, "--dependent-keepdim": dependent_keepdim,
    }
    if args[0] in passes:
        for filename in args[1:]:
            passes[args[0]](Path(filename))
        sys.exit()
    revision = args.pop(1) if args and args[0] == "--revision" else None
    if revision:
        args.pop(0)
    for filename in args:
        convert(Path(filename), revision)
