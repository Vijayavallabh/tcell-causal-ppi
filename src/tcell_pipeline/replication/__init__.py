"""Replication datasets: scPerturb h5ad -> the DE-statistics matrix the EG-IPG pipeline consumes.

Deliberately standalone. Nothing here imports or mutates the modules a running screening lane has
already loaded (``config``, the encoders, the trainer), so a replication adapter can be built and run
while a GPU campaign is in flight on the reference dataset.
"""
