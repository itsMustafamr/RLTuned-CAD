# Pipeline figure: concept for the RL part of BRepVLM

Draft: `pipeline_concept.png` / `pipeline_concept.svg` (vector, editable in Illustrator, Inkscape or
PowerPoint). Regenerate with `python figures/make_pipeline_figure.py`. The 3D shapes are real chunk-9
models rendered from our compiler, not stock art.

## The story it tells (left to right, top to bottom)

**(a) Data.** One real part shown three ways: the STEP solid, the B-rep code the model must write
(colour-coded: vertices, surfaces, loops, edges), and the text annotation. Message: the model writes
geometry directly as a B-rep, not as a sketch-extrude script like ReCAD.

**(b) GRPO loop.** Prompt → BRepVLM policy → 8 sampled B-reps → BRep compiler (OpenCascade) →
solids with rewards → group-relative advantages → policy update, with the update arrow closing the
loop. Four example outcomes make the reward tangible: a correct solid (0.97), a roughly right one
(0.62), a valid but wrong shape (0.33), and a broken one that only exists as edges (0.21). The dashed
coral box is ReCAD-style GT guidance on hard prompts (phase 2).

**(c) Verifiable reward.** The reward drawn as a ladder: no output, then parse and topology, then edges
match, then a valid solid scored by min(IoU, Chamfer) plus structure. Next to it, the sanity check
bars: GT = 1.00 and every corruption scores lower, so reviewers can see the reward is well-behaved
before any training curve.

**(d) Curriculum and data budget.** The curriculum as one coloured bar (planar → planes + cylinders →
other surfaces → complex) with the two answer-length stages, plus the data funnel from 10,000 ABC
models to 2,877 RL prompts.

## For the paper version

* Keep (a) and (b) as the main figure (ReCAD's Fig. 2 has the same two-band layout); (c) and (d)
  can become a second figure or move to the supplement.
* Once results exist, replace the four example rollouts in (b) with real samples from one test prompt,
  and add a strip of before / after GRPO outputs on the same prompts (ReCAD's Fig. 4 style).
* Same palette across figures: purple = model / training, teal = compiler + reward, amber = data and
  curriculum, coral = guidance and failures.
* For image-to-sequence, add a rendered image next to the text prompt in (b); nothing else changes.
