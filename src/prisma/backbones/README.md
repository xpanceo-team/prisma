# FairChem legacy model runtime

This package contains the GemNetT and EquiformerV2 runtime code required by
PRISMA. It was derived from `fairchem-core` 1.10.0 at commit
`977a80328f2be44649b414a9907a1d6ef2f81e95` and is used under the MIT license
included in this directory.

Training infrastructure, datasets, calculators, command-line tools, and other
FairChem models are intentionally excluded. PRISMA-specific behavior remains in
the wrappers under `prisma.models.gnns`.
