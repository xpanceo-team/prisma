# DFT/VASP validation

The `vaspoperator` package runs dependency-aware VASP workflows on a
SLURM-managed cluster. It supports structure relaxation, self-consistent field
calculations, dielectric response, density of states, and band structures.

Install the Python dependencies:

```bash
python -m pip install -e ".[dft]"
```

The workflow also requires a licensed VASP installation, configured SLURM
execution environment, and VASP pseudopotentials accessible to pymatgen:

```bash
export PMG_VASP_PSP_DIR="/path/to/potcars"
```

Configure the site-specific execution and calculation settings in:

- `config/server.yaml`
- `config/vasp.yaml`
- `config/steps.yaml`
- `config/sumo.yaml`

Run one structure:

```bash
python src/vaspoperator/pipelines/run_single_structure.py \
    --structure_path="data/raw/POSCAR_sample" \
    --material_id="Ag2O_test"
```

Run a Parquet batch containing JSON-serialized pymatgen structures:

```bash
python src/vaspoperator/pipelines/run_multi.py \
    --dataset_path="data/raw/test_structures.parquet" \
    --structure_column="structure_mattersim" \
    --id_column="material_id" \
    --num_threads=5 \
    --limit=3
```

The backend writes VASP work directories under `data/vasp/` and aggregated
results under `data/results/` or `data/results_batch/`.
