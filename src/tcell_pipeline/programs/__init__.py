from tcell_pipeline.programs.program_basis import (
    fit_program_basis,
    load_program_basis,
    load_zscore_rows,
    save_program_basis,
    save_program_response,
    train_row_indices,
)
from tcell_pipeline.programs.program_decoder import ProgramDecoder

__all__ = [
    "ProgramDecoder",
    "fit_program_basis",
    "load_program_basis",
    "load_zscore_rows",
    "save_program_basis",
    "save_program_response",
    "train_row_indices",
]

# NO BLAS/OpenMP thread caps here, deliberately. Setting them in this __init__ does make them precede
# numpy for `-m tcell_pipeline.programs.run_basis_study` — but this package is imported by model.py,
# training/dataset.py, run_train.py, run_stage_b.py and screening/run_screening.py, so it would pin
# OMP_NUM_THREADS on every training and screening run, including the 20-epoch GPU campaigns. A one-cell
# fix must not throttle the project. The cap belongs in the LAUNCH COMMAND; run_basis_study reports
# whether it actually took effect rather than assuming it did.
