# Output format

The output JSON is a flat schema (no nested fragments).

## High-level shape

- `version`: output schema version
- `source`: input file info
- `partitioning`: algorithm + requested fragment count
- `chemical_system.fragments`: list of fragments

See also: the [Python API](python_api.md) for the data model.
