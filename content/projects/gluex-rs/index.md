+++
title = "gluex-rs"
+++
<!-- markdownlint-disable MD034 -->
{{ project_header(project="gluex-rs") }}

[`gluex-rs`](https://github.com/denehoffman/gluex-rs) collects tools for GlueX analyses in a single Rust workspace with a unified Python package. It covers shared physics constants and run-period metadata, database access, luminosity calculations, generated-event output, and command-line utilities.

## Workspace

- `gluex-core` provides shared physics constants, run-period metadata, histogram helpers, and serialization primitives.
- `gluex-ccdb` is a read-only CCDB client with typed column accessors and caching.
- `gluex-rcdb` provides an RCDB query layer with expression builders for run selection.
- `gluex-lumi` combines CCDB and RCDB payloads for luminosity calculations.
- `gluex-rs` re-exports the Rust APIs and owns HDDM generation support and the `gluex` command-line interface.
- The Python package exposes the same core, CCDB, RCDB, luminosity, and generation tools through PyO3.

## Analysis workflow

The Python package can write accepted `laddu.GeneratedBatch` values to GlueX HDDM. `laddu` remains responsible for defining reactions, generating events, and performing rejection sampling; `gluex-rs` handles GlueX particle mapping and HDDM output.

The `gluex` command-line tool also exposes luminosity calculations and run-period metadata. Luminosity results are emitted as JSON so they can be passed directly into downstream analysis steps.
