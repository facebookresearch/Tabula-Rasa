# Third-Party Notices

Tabula Rasa incorporates third-party material. Each item below lists what is
used, where it lives, and the applicable licence or citation.

---

## 1. Threefry2x32 — Random123 and JAX

**Used in:** `webgpu/shaders/noise.wgsl` (`threefry2x32`, `getRotation`,
`SKEIN_KS_PARITY`)

Algorithm from J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw,
"Parallel Random Numbers: As Easy as 1, 2, 3", *Proceedings of SC'11*.

### JAX — Apache-2.0

    Copyright 2021 The JAX Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

A complete copy of the Apache License 2.0 is included at
[`third_party_licenses/Apache-2.0.txt`](third_party_licenses/Apache-2.0.txt).

**Statement of modifications.** `threefry2x32`, `getRotation` and
`SKEIN_KS_PARITY` in `webgpu/shaders/noise.wgsl` are a translation into WGSL of
the Threefry-2x32 implementation in JAX's `jax/_src/prng.py`
(`_threefry2x32_lowering` and `rotate_list`). The changes are: translation from
Python/NumPy to WGSL while retaining a 20-round loop, and removal of everything
specific to JAX's tracing and lowering machinery.

### Random123 — BSD-3-Clause

```
Copyright 2010-2011, D. E. Shaw Research. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions, and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 2. PCG hash function — Apache-2.0 (M. E. O'Neill)

**Used in:** `webgpu/shaders/noise.wgsl` (`pcgHash` and dependants)

```
Copyright (c) 2014 M. E. O'Neill / pcg-random.org

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0
```

Upstream: PCG, M. E. O'Neill, <https://www.pcg-random.org>. A complete copy of
the Apache License 2.0 is included at
[`third_party_licenses/Apache-2.0.txt`](third_party_licenses/Apache-2.0.txt).

**Statement of modifications.** Translated from GLSL to WGSL (`uint` to `u32`)
and renamed to `pcgHash`. The constants, shift schedule and operation order are
unchanged.

---

## 3. Larcher-Pillichshammer / Sobol radical inverses — BSD-2-Clause (pbrt-v2)

**Used in:** `python/tabula_rasa/lpgk_sampler.py` (`larcher_pillichshammer_riu`,
`gruenschloss_keller_riu`)

**Upstream:** pbrt-v2, `src/core/montecarlo.h` (`LarcherPillichshammer2`,
`Sobol2`) — <https://github.com/mmp/pbrt-v2>

```
pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**Statement of modifications.** `larcher_pillichshammer_riu` reproduces the loop
body of pbrt's `LarcherPillichshammer2` — initial `v = 1 << 31`, the
`v |= v >> 1` column recurrence, `n >>= 1`, and `if (n & 1) scramble ^= v`.
`gruenschloss_keller_riu` applies the same technique with pbrt's `Sobol2`
recurrence `v ^= v >> 1`, changed to the Gruenschloss-Keller third-dimension
matrix (initial `3 << 30`, accumulating `v2 << 1`). Both are rewritten as a
32-iteration `jax.lax.scan` rather than a `for` loop, and both return
`scramble * 2^-32` over the full 32 bits instead of pbrt's 24-bit
`min(((scramble >> 8) & 0xffffff) / float(1 << 24), OneMinusEpsilon)`.

Underlying constructions:

- G. Larcher and F. Pillichshammer, "Walsh series analysis of the L2-discrepancy
  of symmetrisized point sets", *Monatshefte für Mathematik* 132, 1–18, 2001.
- L. Gruenschloss and A. Keller, "(t,m,s)-Nets and Maximized Minimum Distance,
  Part II", in P. L'Ecuyer and A. Owen (eds.), *Monte Carlo and Quasi-Monte
  Carlo Methods 2008*, Springer-Verlag, 2009.

---

## 4. Kensler permutation

**Used in:** `python/tabula_rasa/lpgk_sampler.py` (`permute`)

Hash-based index permutation adapted from A. Kensler, "Correlated
Multi-Jittered Sampling", Pixar Technical Memo 13-01, 2013.

---

## Classical methods

Implemented from published descriptions; listed for citation.

| Method | Location | Source |
|---|---|---|
| van der Corput / Hammersley radical inverse | `python/tabula_rasa/hammersley_sampler.py`, `webgpu/shaders/noise.wgsl` | van der Corput (1935); Hammersley (1960) |
| Box–Muller transform | `webgpu/shaders/noise.wgsl` | Box & Muller, *Ann. Math. Statist.* 29(2), 1958 |
| Thin-lens depth of field, polar disk warp | `python/tabula_rasa/raytracing_map.py` | Pharr, Jakob & Humphreys, *Physically Based Rendering*, §6.2.3, §13.6.2 |
| Linear-blend skinning | `webgpu/shaders/rasterizer_mesh.wgsl`, `webgpu/main.js` | glTF 2.0 specification, "Skins" |
| Quaternion spherical linear interpolation | `webgpu/gltf-loader.js` (`slerpQuaternion`) | Shoemake, *SIGGRAPH '85*. Uses the widely reproduced shortest-arc form: negate one input on a negative dot product, and fall back to normalised linear interpolation above a `0.9995` dot threshold. |
| Quaternion-to-matrix TRS composition, node hierarchy accumulation | `webgpu/gltf-loader.js` (`composeTRSMatrix`, `computeNodeWorldMatrices`) | glTF 2.0 specification, "Transformations". The matrix entries are the fixed algebraic expansion of a unit quaternion; column-major to match WebGPU. |
| Linear Counting cardinality estimator | `python/tabula_rasa/count_unique_estimator.py` (`count_linear`) | Whang, Vander-Zanden & Taylor, *ACM TODS* 15(2), 1990. The estimate is the closed form `-m·ln(V/m)`. Currently unreachable — only `count_histo` is returned. |
| Cranley-Patterson rotation | `python/tabula_rasa/hammersley_sampler.py` (currently disabled) | Cranley & Patterson, *SIAM J. Numer. Anal.* 13(6), 1976. Standard randomised-QMC shift: add a uniform per-dimension offset and take the fractional part. |

Legacy LCG multipliers appear as bare constants in
`webgpu/shaders/noise.wgsl`: `1664525` (*Numerical Recipes*) and `0x9E3779B9`
(golden-ratio / TEA delta, Knuth).
