# Equal-margin lock spec (four-side gap)

This is the numeric definition of "检测到机台让它缩放到中间，左右上下比例一直一样":
when the machine is locked, the screen sits in the middle of the frame and the
gap between the play field and the button ring is the same on the left, right,
top and bottom, and stays that way over time.

The operator-facing explanation of the same geometry lives in the README under
`## 拉回正面（--ring-round）`; this file is the acceptance contract and the
measurement that decides pass or fail.

## What is measured

Everything is measured on the **delivered** (rectified, locked) pixels, not on
the renderer's internal state. Two reasons:

* the live pipeline models the screen as a circle, so its `geometry_margins`
  field is left == right and top == bottom on every frame by construction and
  carries no information about how the machine landed;
* the renderer knows where it *tried* to put the machine, so asking it where the
  machine ended up proves nothing.

Per sampled frame:

1. find the play field (`inner_screen`); its centre `c` and radius `r` are the
   box centre and the mean box half-size;
2. find the eight purple button blobs around it and take their centroids
   (`pc/canonical.purple_points`, the one finder the live path and this tool
   share);
3. for each of the eight slots, `rho_slot = |button - c| / r`, angles 22.5 deg
   apart with slot 1 at 22.5 deg (`pc/canonical.SLOT_ANGLES`);
4. collapse the slots to four sides (`pc/canonical.SIDE_SLOTS`):
   `left = (slot 3, 4)`, `right = (slot 7, 0)`, `top = (slot 1, 2)`,
   `bottom = (slot 5, 6)`.

The gap in millimetres is `(rho_slot - 1) * r_mm`, so `rho = 1` is a button
touching the screen and the real 75 mm gap is `rho = 1 + 75 / r_mm`.

A second, independent number guards the first: strip the screen out of the
measurement entirely and ask whether the eight button centroids, about their own
mean, form a circle. That is `ring_aspect`, the minor/major ratio of their
covariance. It shares no input with `rho` except the button points, so a
correction cannot satisfy the spec ratio by squashing the ring.

## Target

`rho = TILE_RATIO` (default `1.23`) in **every** direction. The number is not
the physical 75 mm ratio: it is measured in detector units, where the cyan play
field stands in for the screen and the button colour centroid stands in for the
button centre. Hand-measuring the same reference photo against the white bead
ring gives 1.365, 9 % higher, so the denominator must stay paired with the
detector and must not be mixed. Re-derive it with
`tools/head_on_ratio.py` on a square-on reference photo if the detector changes.

The eight slot radii being equal is the whole condition; a circular screen alone
does not imply it, and that gap is exactly what the operator sees.

## Acceptance

| metric | definition | bar |
| --- | --- | --- |
| spread | `(max - min) / mean` over the four side `rho` values | `<= 0.08` per frame, `<= 0.05` stretch |
| frames passing | fraction of measurable frames with `spread <= 0.08` | `>= 0.95` |
| ring aspect | minor/major of the eight button centroids about their own mean, no screen reference | `>= 0.95` median |
| L/R imbalance | `abs(left - right) / mean` | `<= 0.03` median |
| T/B imbalance | `abs(top - bottom) / mean` | `<= 0.03` median |
| steadiness | temporal std of each side `rho` | `<= 0.02` |
| centring | play-field centre offset from frame centre | `<= 2 %` of the width |

The 0.08 spread bar is the annotator's own `--gap-tolerance`
(`tools/annotate_geometry.py`), so the labelling convention and the delivery
bar agree. The aspect bar is a floor, not a target: the raw footage sits at
0.978 and the lock should not be the thing that pushes it down.

## Current status (2026-10-07)

Measured with `tools/check_margins.py --every 6`. `L/R/T/B` are the four side
`rho` values, `spread` is the per-frame `(max - min) / mean`, `aspect` is the
independent ring check.

| clip | L | R | T | B | spread (median) | aspect | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sessions/20261007-195409/raw.mp4` (input, unlocked) | 1.15 | 1.15 | 1.11 | 1.18 | 6.7 % | 0.978 | - |
| `sessions/20261007-195409/processed-live.mp4` (live) | 1.20 | 1.19 | 1.10 | 1.30 | 17.2 % | 0.948 | FAIL |
| `work/follow_1007/final_1007.mp4` (offline) | 1.24 | 1.26 | 1.13 | 1.35 | 17.3 % | 0.925 | FAIL |

Read as millimetres the bottom gap is about 2.6x the top gap, which is the
"上下比例不一样" the operator reports. Decomposing the eight slot radii by
harmonic shows the two separate defects:

```text
                       raw spread   after removing tilt (order 1)   after order 3
raw.mp4                    7.6 %                 1.7 %                  0.1 %
processed-live.mp4        20.8 %                 5.4 %                  0.4 %
final_1007.mp4            21.9 %                 6.6 %                  0.6 %
```

The **order-1** term (the button ring centre does not coincide with the play
field centre) dominates and is *larger* after the lock than before it; that is
what breaks left/right and top/bottom equality. A 5-7 % **order-2+** residue
survives after the tilt term is removed, which is a genuine two-cycle squash of
the ring around the screen.

## How to reproduce

```powershell
.\.venv\Scripts\python.exe tools\check_margins.py `
  "sessions\20261007-195409\processed-live.mp4" `
  "work\follow_1007\final_1007.mp4" `
  --every 6 --json work\margin1007\report.json
```

`tools\check_margins.py` is the reference implementation of this spec. It also
prints a box-gap cross-check in millimetres (the annotator's
`outer_buttons` - `inner_screen` convention). That cross-check is deliberately
**not** part of the verdict: on locked frames the current v5 outer box swallows
the cabinet body below the ring (bottom 138-145 mm against top 32-45 mm), so it
overstates the defect and cannot be used as the acceptance number.

## Fix and what it buys

The correction already exists and is off in the live path:
`pc/canonical.pull_back_maps` walks every direction back onto `TILE_RATIO` with
a ramp that is zero at the screen edge and one at the ring, so the screen never
moves and the ring lands on one circle. `tools/ring_round_clip.py` applies it to
an already-locked clip so the gain can be settled before it goes into the
renderer.

Post-hoc on `work/follow_1007/final_1007.mp4` (685 frames):

| correction | L | R | T | B | spread (median) | aspect | dead-on | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1.24 | 1.26 | 1.13 | 1.35 | 17.3 % | 0.925 | 1 % | FAIL |
| one pass, gain 1.0 | 1.25 | 1.24 | 1.18 | 1.26 | 7.6 % | 0.950 | 56 % | FAIL |
| one pass, gain 1.8 | 1.26 | 1.22 | 1.23 | 1.21 | 4.6 % | 0.962 | 90 % | FAIL |
| two passes | 1.25 | 1.23 | 1.21 | 1.24 | 3.8 % | 0.960 | 96 % | PASS |

The correction moves **both** numbers the right way: the spec spread falls from
17.3 % to 3.8 % and the independent aspect rises from 0.925 to 0.960. A unity
-gain pass removes only a little over half of the measured error, because the
ramp is evaluated per pixel and the near and far halves of the same button are
pushed by different amounts, so the blob centroid moves less than the ramp
predicts. Gain ~1.8 in one pass, or the same correction twice, closes it, which
makes the correction a closed loop with a per-pass gain of roughly 0.55.

Three things have to hold for this to work in the live path, and getting any one
of them wrong produces the "stretched picture, ring worse" failure:

1. **The map has to be rebuilt every frame.** Centre, screen radius and the ring
   fit all move as the operator moves. A single map built once and replayed is
   anchored to one frame's geometry and drags every other frame; the whole error
   is in the per-frame order-1 term, and the static map cannot see it.
2. **The radius and the ramp have to live in the same space.** `screen_radius`
   and the pixels the map rewrites must be the delivered-frame quantities. If
   the radius comes from the pre-lock detection while the map lands after the
   lock, the ramp is mis-scaled by the lock zoom (this is the ~15 % mismatch),
   and the correction acts as a global zoom instead of a local pull.
3. **Gain above one, or a closed loop.** At unity the loop only removes about
   half the error per pass, so the delivered spread stalls around 7-8 %, which
   still fails the 0.08 bar and leaves visible stretch to show for it.

For the record, the two purple masks in the tree (`pc/canonical.purple_points`
and `pc/machine_lock.measure_button_ring`) were checked against each other on
the same frames: they choose the same eight buttons with a median centroid
offset of about 2 px, so the colour thresholds are not the source of the scale
mismatch.

```powershell
.\.venv\Scripts\python.exe tools\ring_round_clip.py `
  "work\follow_1007\final_1007.mp4" "work\margin1007\final_1007_fixed.mp4" `
  --every 6 --gain 1.8
.\.venv\Scripts\python.exe tools\check_margins.py `
  "work\margin1007\final_1007_fixed.mp4" --every 6
```

Caveat, visible in `work/margin1007/before_after_margin.jpg`: the pull-back
warps the whole outer region, not just the button ring, so the cabinet body and
the background around it are stretched along with the ring (the per-pixel factor
is clamped to +-18 %, and the screen never moves). That is the price of making
the ring concentric; if the stretched cabinet edge reads worse than the uneven
gaps, restrict the remap to the cabinet region instead of the whole frame.
