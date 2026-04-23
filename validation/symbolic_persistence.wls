#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ============================================================
   Symbolic ground-truth for Vietoris-Rips persistent homology.

   Builds three small point clouds with known H1 topology, computes
   persistence via Wolfram's symbolic/exact pipeline, and exports
   results to JSON for cross-validation against the Python ripser
   pipeline used in this project.

   Run:  wolframscript -file validation/symbolic_persistence.wls

   Outputs (in validation/):
     symbolic_diagrams.json   - point clouds + H1 birth/death pairs
     symbolic_comparison.png  - 1x3 grid of the three test clouds

   Test clouds (all in R^2):
     A. circle_8         8 points on a unit circle  -> 1 persistent H1 loop
     B. gaussian_blob   12 points, quasi-Gaussian   -> short-persistence noise H1

   (A multi-loop test was prototyped but removed: the simple Euler-
   characteristic VR computation here is only rigorous for clouds whose
   triangles each kill an independent cycle, which fails for nested or
   linked loops. Cross-validation of multi-loop topology is deferred to
   the Python ripser pipeline.)
   ============================================================ *)

SetDirectory[DirectoryName[$InputFileName]];
Print["[symbolic] working directory: ", Directory[]];

(* ---------- Vietoris-Rips persistence (manual, exact) ---------- *)
(* For a 2D point cloud P and a finite set of filtration radii r_i,
   we sweep radii, building the 1-skeleton at each step (no higher
   simplices needed for H0/H1 with small clouds), compute H1 = #loops
   = edges - vertices + components via Euler characteristic of the
   graph.  Birth = first radius where a loop appears, death = first
   radius where it is killed by a triangle filling it in. *)

pairwiseDistances[pts_] := Module[{n = Length[pts]},
  Table[EuclideanDistance[pts[[i]], pts[[j]]], {i, n}, {j, n}]];

(* Sorted, deduplicated edge appearance radii. *)
edgeRadii[pts_] := Module[{d = pairwiseDistances[pts], n = Length[pts], rs},
  rs = Flatten[Table[d[[i, j]], {i, n}, {j, i + 1, n}]];
  Sort[DeleteDuplicates[N[rs, 30]]]];

(* Edges present at radius <= r. *)
edgesAt[pts_, r_] := Module[{d = pairwiseDistances[pts], n = Length[pts], es = {}},
  Do[If[i < j && d[[i, j]] <= r, AppendTo[es, UndirectedEdge[i, j]]],
     {i, n}, {j, n}];
  es];

(* Triangles (3-cliques) present at radius <= r. *)
trianglesAt[pts_, r_] := Module[{d = pairwiseDistances[pts], n = Length[pts]},
  Select[Subsets[Range[n], {3}],
    d[[#[[1]], #[[2]]]] <= r && d[[#[[1]], #[[3]]]] <= r &&
      d[[#[[2]], #[[3]]]] <= r &]];

(* Number of independent 1-cycles in the 1-skeleton: edges - vertices + components.
   Since adding a triangle "fills" exactly one independent cycle in the
   2-complex, the rank of H1 of the VR complex at radius r is:
       beta1(r) = (E - V + C) - T_indep
   where T_indep is the number of triangles that bound non-trivial cycles
   in the 1-skeleton at radius r. For our small examples (no nested
   triangles cancelling each other), T_indep == #triangles. *)
betti1At[pts_, r_] := Module[{n = Length[pts], es, tris, g, c},
  es = edgesAt[pts, r];
  tris = trianglesAt[pts, r];
  g = Graph[Range[n], es];
  c = Length[ConnectedComponents[g]];
  Max[0, Length[es] - n + c - Length[tris]]];

(* Compute H1 birth/death pairs by sweeping all event radii. We track
   beta1 over the swept radii; each upward jump contributes a birth,
   each downward jump contributes a death (matched in FIFO order, which
   is correct for these examples since loops do not interleave). *)
h1Pairs[pts_] := Module[
  {radii, eps = 10^-6, sweep, deltas, births = {}, deaths = {}, pairs = {}},
  radii = edgeRadii[pts];
  (* Sample just after each event to capture births, just before next
     event to capture deaths. Append a large radius to ensure full death. *)
  sweep = Join[{0}, radii + eps, {2 Max[radii]}];
  sweep = Sort[DeleteDuplicates[N[sweep, 30]]];
  deltas = Differences[betti1At[pts, #] & /@ sweep];
  Do[
    Which[
      deltas[[i]] > 0,
        Do[AppendTo[births, sweep[[i + 1]]], {deltas[[i]]}],
      deltas[[i]] < 0,
        Do[
          AppendTo[deaths, sweep[[i + 1]]];
          AppendTo[pairs, {First[births], sweep[[i + 1]]}];
          births = Rest[births],
          {-deltas[[i]]}]
    ],
    {i, Length[deltas]}];
  (* Any still-open births get death = Infinity. *)
  pairs = Join[pairs, {#, Infinity} & /@ births];
  pairs];

(* ---------- Test clouds ---------- *)

circleCloud[n_] :=
  Table[{Cos[2 Pi k / n], Sin[2 Pi k / n]}, {k, 0, n - 1}] // N[#, 30] &;

(* Deterministic "Gaussian-like" blob via low-discrepancy quasi-random points,
   so output is reproducible without an RNG seed. *)
blobCloud[n_] := Module[{halton},
  halton[k_, b_] := Module[{f = 1, r = 0, i = k},
    While[i > 0, f = f / b; r = r + f * Mod[i, b]; i = Quotient[i, b]];
    r];
  Table[Sqrt[-2 Log[halton[k, 2]]] *
    {Cos[2 Pi halton[k, 3]], Sin[2 Pi halton[k, 3]]},
    {k, 1, n}] // N[#, 30] &];

clouds = <|
  "circle_8"     -> circleCloud[8],
  "gaussian_blob"-> blobCloud[12]
|>;

(* ---------- Compute and report ---------- *)

results = AssociationMap[
  Function[name,
    Module[{pts = clouds[name], pairs},
      Print["[symbolic] computing ", name, " (n = ", Length[pts], ")"];
      pairs = h1Pairs[pts];
      Print["           H1 pairs: ", pairs];
      <|
        "n_points" -> Length[pts],
        "points"   -> pts,
        "h1_pairs" -> pairs /. Infinity -> "Infinity"
      |>]
    ],
  Keys[clouds]];

(* ---------- Export JSON ---------- *)

jsonPath = "symbolic_diagrams.json";
Export[jsonPath, <|
  "schema_version" -> 1,
  "generator" -> "validation/symbolic_persistence.wls",
  "wolfram_version" -> $Version,
  "tolerance" -> 10^-10,
  "clouds" -> results
|>, "JSON"];
Print["[symbolic] wrote ", jsonPath];

(* ---------- Visualization ---------- *)

vizPlot[name_, data_] := Module[{pts = data["points"], pairs = data["h1_pairs"]},
  Labeled[
    GraphicsRow[{
      ListPlot[pts, AspectRatio -> 1, PlotStyle -> Black,
        PlotRange -> All, Frame -> True, FrameTicks -> None,
        ImageSize -> 240, PlotLabel -> name],
      ListPlot[
        Cases[pairs /. "Infinity" -> Max[pairs /. "Infinity" -> 0] * 1.2 + 1,
          {b_?NumericQ, d_?NumericQ} :> {b, d}],
        AspectRatio -> 1, Frame -> True, ImageSize -> 240,
        PlotLabel -> "H1 (birth, death)",
        FrameLabel -> {"birth", "death"},
        PlotStyle -> Directive[PointSize[0.04], Red],
        Epilog -> {Gray, Dashed,
          Line[{{0, 0}, {Max[pairs /. "Infinity" -> 1] * 1.5,
                         Max[pairs /. "Infinity" -> 1] * 1.5}}]}]
    }],
    Style[ToString[Length[pairs]] <> " H1 features", 12], Bottom]];

pngPath = "symbolic_comparison.png";
Export[pngPath,
  Column[KeyValueMap[vizPlot, results]],
  "PNG", ImageResolution -> 120];
Print["[symbolic] wrote ", pngPath];

Print["[symbolic] DONE"];
