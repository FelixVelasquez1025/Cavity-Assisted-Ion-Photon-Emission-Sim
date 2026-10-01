# Repository comparison and portfolio cleanup

Reviewed local checkouts on September 30, 2026. No remote fetch was needed for
this comparison; conclusions refer to the histories and working files present
on disk.

| Checkout | Checked-out branch | Last commit | Role |
| --- | --- | --- | --- |
| CAIPE-sim | main | b3475d4, August 27, 2025 | Original three-level cavity project |
| CAIPE-sim-1 | Yb171_Rabi_Flopping | 8728bcf, February 2, 2026 | Latest Yb-171 development |

The older HEAD is an ancestor of the newer HEAD. There are 26 commits in
`b3475d4..8728bcf`, including merges. Every tracked file in the old checkout
exists in the newer checkout. After normalizing CRLF line endings, 13 of the
17 old tracked files are identical; only README, project metadata, gitignore,
and the lockfile differ. No unique old implementation needs to be ported.

The old checkout's apparent modifications were entirely line endings. The newer
checkout also contained real edits to the three-level example, batch runner,
Hamiltonian helpers, and other files, plus untracked examples, notes, and
results. These are valuable working files, not evidence that the old clone
should replace the new one. Both directories and their histories are preserved.

The newer remote is on a private Forgejo host; the older remote is the original
GitHub project. The latest work is on the Yb branch, not local main. A future
portfolio publication must use the correct branch and preserve attribution
to the original collaborative project.

## First cleanup pass

- Promoted the latest pulse-enabled example into the installable package and
  retained its example path as a launcher.
- Used the already-present local Yb171 data module instead of requiring a
  custom atomic-physics checkout in a neighboring directory.
- Replaced the QuTiP Options import with a supported options dictionary.
- Added a CPU CLI demo that exports populations, pulse envelopes, and cavity
  observables as PNG and NPZ, plus numerical and analytic checks.
- Separated runtime dependencies from optional JAX and notebook tools.
- Documented which implementation is active and its remaining model limits.

Validation: eight active tests passed on Python 3.12 with both the frozen
QuTiP 5.2.0 environment and a separate QuTiP 5.3.1 installation. The installed
CLI exported PNG/NPZ results, and the interactive application initialized four
figures using a headless Matplotlib backend. Desktop window interaction and
optional JAX/GPU execution were not exercised.

## Remaining portfolio work

The [review against the seven-week plan](plan-review.md) incorporates the
foundational Travers Ward thesis and the user's confirmation that the Yb
numerical inputs were independently verified.

1. Complete the model-to-thesis mapping and photon-output analysis described
   in [model-status.md](model-status.md) and [plan-review.md](plan-review.md).
2. Merge the older batch/UI helpers into the pulse-enabled implementation and
   extract the solver from the Matplotlib UI. Retire duplicate implementations
   only after checking feature parity.
3. Add a concise explanation of the experiment, a level diagram, and a few
   validated parameter studies with interpretation and convergence evidence.
4. Reconcile historical examples/tests and generated files. The Ca40 example
   has a UTF-8 BOM; several historical examples still need the customized
   atomic-physics dependency.
5. Choose a public repository and license with the original contributors,
   document individual contributions, and publish the reviewed branch.
