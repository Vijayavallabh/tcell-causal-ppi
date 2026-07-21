"""Program-basis package.

The BLAS/OpenMP caps below are set HERE, before this package's own imports, and that placement is the
whole point. ``run_basis_study`` set them at its module top with the comment "must precede the numpy
import" — but running it as ``-m tcell_pipeline.programs.run_basis_study`` imports this package first,
and the line below used to import ``program_basis`` -> numpy, so libgomp and OpenBLAS had already sized
their pools to all 64 cores by the time that line ran. The driver then printed ``OMP=4`` as if the cap
had taken effect. That is the recorded shared-box regression: 64 threads became ~830, load hit 600, and
a 4-minute fit took 87. ``setdefault`` means an explicit env var from the caller still wins.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

from tcell_pipeline.programs.program_basis import (  # noqa: E402
    fit_program_basis,
    load_program_basis,
    load_zscore_rows,
    save_program_basis,
    save_program_response,
    train_row_indices,
)
from tcell_pipeline.programs.program_decoder import ProgramDecoder  # noqa: E402

__all__ = [
    "ProgramDecoder",
    "fit_program_basis",
    "load_program_basis",
    "load_zscore_rows",
    "save_program_basis",
    "save_program_response",
    "train_row_indices",
]
