# Discovery notebooks

Use notebooks for exploration, profiling, and visual checks. Production
transformations must live in importable modules under `src/` and be callable
without executing notebook cells.

The source-discovery notebooks are `syndrome_data_discovery.ipynb`,
`qasmbench_data_discovery.ipynb`, and `google_qec_data_discovery.ipynb`.
Automated checks belong in `tests/*.py` and run with `make test`.

Recommended first notebook sections:

1. Bronze object and archive-member inventory;
2. schema/type profile per source;
3. missingness, duplicates, ranges, and structural invariants;
4. candidate entities, keys, and relationship cardinalities;
5. evidence for rejected cross-source joins.

Clear large cell outputs before committing notebooks.
