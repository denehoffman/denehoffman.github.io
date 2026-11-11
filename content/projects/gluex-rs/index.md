+++
title = "gluex-rs"
+++
{{ <project_header project="gluex-rs" /> }}

[`gluex-rs`](https://github.com/denehoffman/gluex-rs) is one unified library for GlueX analysis, with a Rust API, a Python module named `gluex`, and the `gluex` command-line program. Run metadata, calibration constants, photon flux, luminosity, event files, and standalone event generation share the same core model.

## One connection, typed queries

Open RCDB and CCDB SQLite snapshots once, select runs with lazy typed queries, and collect immutable results. Results retain their run selection, calibration time, variation, omissions, and source paths, making it easier to trace an analysis back to its inputs.

```python
import gluex

gx = gluex.connect(rcdb="rcdb.sqlite", ccdb="ccdb.sqlite")
beam_current = gx.runs.conditions["beam_current"]
runs = (
    gx.runs.select([gluex.RunPeriod("2018-01"), gluex.RunPeriod("2018-08")])
    .where(gx.runs.aliases.is_coherent_beam & (beam_current > 10.0))
    .collect()
)
print(runs.numbers)
print(runs.provenance)
```

The Rust crate exposes this model through `GlueX`, `RunPeriod`, `RunSelection`, and `SourceConfig`. Both database snapshots are opened read-only; `RCDB_CONNECTION` and `CCDB_CONNECTION` can supply their paths instead of explicit arguments.

## Analysis tools

- Select runs and conditions from RCDB, and read typed calibration payloads from CCDB.
- Calculate photon flux and luminosity using shared run-period and coherent-energy metadata.
- Read and write AmpTools, FSRoot, and Pawian event files.
- Generate standalone HDDM events and write accepted `laddu.GeneratedBatch` values for GlueX workflows.
- Use the same tools from Python, Rust, or the command line.

See the [getting-started guide](https://github.com/denehoffman/gluex-rs/blob/main/docs/getting-started.md) and [Rust API reference](https://docs.rs/gluex-rs).
