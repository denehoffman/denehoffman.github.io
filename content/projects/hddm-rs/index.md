+++
title = "hddm-rs"
+++
{{ project_header(project="hddm-rs") }}

[`hddm-rs`](https://github.com/denehoffman/hddm-rs) brings the Hierarchical Data Description Model used by GlueX into Rust. The workspace covers runtime I/O, compression and schema handling, derive macros for manually defined models, and a code generator for producing Rust bindings from HDDM headers.

## Workspace

- `hddm` reads and writes records, handles schemas, and supports zlib and bzip2 compression.
- `hddm-derive` provides `HddmRead` and `HddmWrite` derive macros for compatible Rust types.
- `hddm-rs` provides the binding generator and command-line interface.

Generated bindings contain a Rust structure for each field in the HDDM model together with convenience functions for opening, creating, reading, and writing files:

```console
hddm-rs sample_mc.hddm -o hddm_s.rs
```

The generator can also run from a Cargo `build.rs`, allowing bindings to stay synchronized with a model at compile time and then be included from `OUT_DIR`.

This implementation is not a direct port of the original C++ code. It builds on the HDDM format and acknowledges the original work by Richard Jones and other Jefferson Lab contributors.
