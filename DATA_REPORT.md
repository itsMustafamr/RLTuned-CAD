# What is in the data (chunk 9) and how we use it

Source: the BRepVLM share from Bhavika (Dropbox), ABC chunk `abc_0009_step_v00`.
Everything below was measured on that chunk with the code in this repo
(`scripts/prepare_data.py`, `brepgrpo/`), not estimated.

## 1. What we received

| Item | File | Content |
|---|---|---|
| Sequences | `Sequences_chunk9/seq_12bit_chunk9.tar.gz` (27 MB) | 4,477 `.seq.txt` files, v3 protocol, 12-bit, split into `simple/` (3,802) and `complex/` (675), plus per-chunk normalization and quantization diagnostics |
| Text annotations | `Text_Annotations_chunk9/abc_0009_step_v00.tar.gz` (3.6 MB) | 5,995 model folders, 3 descriptions each: `construction_steps`, `global_shape`, `topology_relations` |
| STEP files | `Step_files_chunk9/abc_0009_step_v00.7z` (1 GB) | the original ABC STEP files (not needed for training) |

Not received yet: the BRep compiler / reconstruction code, the metadata JSONs behind the
annotations, image renders for image-to-sequence, and the SFT checkpoint.

## 2. How the 10,000 raw models became 4,477 sequences

From the generator log (`generation.log`):

| Outcome | Models |
|---|---|
| sequence written | 4,477 |
| skipped: more than one solid | 2,669 |
| skipped: more than 70 faces | 1,078 |
| skipped: source fidelity budget | 987 |
| skipped: unsupported surface (extrusion / revolution / offset surfaces) | 586 |
| skipped: wire canonicalization / quantization collapse | 189 |
| errors + timeouts | 14 |

`simple` means up to 29 faces, `complex` means 30 to 70 faces.

## 3. The sequence format (what the model has to write)

A pure B-rep, written as code, every number a 12-bit integer:

```
model(3)
solid(0, shells=[0])
shell(0, faces=[0, 1, 2, 3, 4, 5])
v(0,1303,2540,0)                                    vertices
f0 = plane(0, n=[2047, 4094, 2047], d=2540, centroid=[2047, 2540, 2047])
wire(0, face=0, r=0, edges=[-1, -2, 3, 4])          loops: signed, 1-based edge refs; r=0 outer, r=1 hole
e0 = line(0, v=[0,1])                               edges: curve + start/end vertex
end_model()
```

* Surfaces: plane 48.5k, cylinder 22.9k, cone 2.1k, torus 1.7k, sphere 0.6k, B-spline surface 0.3k
* Curves: line 117k, circle 52k, B-spline 16k, ellipse 1.3k, degenerate `edge_other` 0.8k
* Dequantization: `x = -1 + 2 * q / 4094` for coordinates and directions, `r = 2 * q / 4094` for radii,
  cone `semi_angle` is in whole degrees. The part sits in a [-1, 1] box (longest side = 2).

Conventions we had to work out (they are not written down anywhere, so worth confirming with Bhavika):

* circle / ellipse `sweep`: `0` = the arc runs counter-clockwise about `axis_dir` from `v[0]` to `v[1]`,
  `1` = clockwise, `2` = full curve. Reading it this way rebuilds 297 of 300 test models into valid
  solids; ignoring it rebuilds only 213 of 300.
* wire edge references are 1-based and signed (sign = direction of travel).
* degenerate `edge_other` edges are the only edges used by one loop instead of two (all 795 of them).

## 4. Length is the main cost

Token counts with the Qwen tokenizer rules (every digit is its own token), a slight lower bound:

| | median | 90th pct | share that fits in 8k | share that fits in 16k |
|---|---|---|---|---|
| simple | 2.2k | 5.5k | 97% | 99.5% |
| complex | 10.0k | 16.0k | 29% | 91% |
| all | 2.7k | 9.0k | 87% | 98% |

One "simple" model (`00094035`) is 2.8 MB / 2.5M tokens and is dropped; in total 120 shapes are longer
than the 15k-token budget and are left out of SFT and RL. RL runs a short-answer stage (8k) first and a
long-answer stage (16k) second, which is also our curriculum.

## 5. Ground-truth quality checks (all pass)

On every one of the 4,476 sequences:

* parses cleanly, every face/edge/vertex reference is valid
* every loop closes when you chain its edges vertex to vertex
* every non-degenerate edge is used by exactly two loops (closed 2-manifold)

Our stand-in compiler (`brepgrpo/compiler_occ.py`, OpenCascade + ShapeFix) rebuilds **3,805 of the
3,856 unique annotated sequences (98.7%)** into valid closed solids (median 28 ms per model). Where the annotation states a volume, the
rebuilt solid matches it with a median error of **0.1%** (93% within 2%), and surface area matches
with a median error of 0.04%. So the sequences, the annotations and our compiler agree.

## 6. The text annotations

Three descriptions per model. They are built from metadata, so they carry checkable numbers:
face / edge / vertex counts (stated in ~68% of `global_shape` texts, and they match the sequence 99%
of the time), normalized dimensions ("length of 2.0 units"), volume, surface area, bounding-box fill,
and the area share of each surface type. They do not pin down everything (for example, exact hole
positions), so one text can fit several shapes. This is why the RL reward also scores face/edge/vertex
counts and surface-type mix, not only overlap with one ground-truth shape.

* 4,375 of 4,477 sequences have annotations (102 do not).
* 1,620 annotation folders have no sequence (models filtered out by the v3 generator).

## 7. Duplicates

Only 3,956 of the 4,477 sequences are unique: 153 groups of exact duplicates cover 674 models (the
largest group has 21 copies). We keep one representative per shape and never let a duplicate group
straddle train and test.

## 8. What the pipeline uses

| Set | Shapes | Notes |
|---|---|---|
| unique shapes with annotations | 3,856 | |
| within the 15k-token budget | 3,736 | 120 longer shapes are left out of SFT and RL |
| ground truth rebuilds into a valid solid | 3,686 of 3,736 | needed for the reward (98.7%) |
| train / val / test split | 3,470 / 116 / 270 | stratified by simple / complex, 90 / 3 / 7 |
| SFT samples | 13,452 train, 452 val | 4 prompts per shape: each description alone + all three combined |
| RL prompts, 8k stage | 2,877 | curriculum order: small planar, planes + cylinders, other surfaces, complex |
| RL prompts, 16k stage | 3,318 | |
| test prompts with rebuildable GT | 255 | used by `scripts/eval.py` |

## 9. Asks for Bhavika

1. **Compiler / reconstruction code.** Until then we use our stand-in. Hers plugs in through
   `BREP_COMPILER_MODULE` (`brepgrpo/compiler.py`); the reward stays comparable.
2. **SFT checkpoint details:** base model and size, the exact prompt template, and whether the target
   includes the `# ...` comment lines and the `quantization(...)` header. We train on the body only,
   which saves about 200 tokens per sample.
3. **Metadata JSONs and image renders** (with the camera setup) for image-to-sequence.
4. **Format suggestions that shorten sequences:**
   * 85% of `bspline` edges are degree 1 with two poles, which are straight segments. Writing them as
     `line(...)` removes 7% of all tokens and a duplicate way to say the same thing.
   * `centroid=[...]` on every face is 7.7% of tokens. Our compiler never reads it and still rebuilds
     98.7% of the models, so it may be droppable (worth checking whether her compiler needs it).
   * unquantized floats (B-spline weights and knots such as `0.804737854124365`) are 5.9% of tokens.
5. **Splits:** please split by duplicate group too, otherwise copies of a shape can end up in both
   SFT training and our RL test set.
6. **More data:** one chunk gives about 3.9k unique usable shapes. If the first RL numbers look good,
   3 to 5 more chunks would be the next step.
