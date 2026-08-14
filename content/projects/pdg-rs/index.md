+++
title = "pdg-rs"
+++
{{ project_header(project="pdg-rs") }}

[`pdg-rs`](https://github.com/denehoffman/pdg-rs) provides programmatic access to the Particle Data Group database of particle-physics measurements. It is independently developed and is not affiliated with the PDG; scientific uses should cite the underlying Review of Particle Physics data.

The crate exposes both a Rust API and a `pdg` command-line program. It works with particle, measurement, reference, and explanatory-text records rather than limiting access to a curated list of particle properties.

## Searching the database

The CLI can look up PDG string identifiers, search particle records, and search the database's text content:

```console
pdg show S008245
pdg show M036 --summary
pdg search particles K --limit 5
pdg search text "form factors"
```

The first command that needs the default SQLite database downloads and verifies it in the operating system's cache directory. Dedicated `pdg db` commands report, fetch, locate, or clear that cache; environment variables and an offline flag support reproducible or disconnected workflows.

## Rust interface

The Rust API can resolve particles by name or Monte Carlo ID, query properties such as masses and lifetimes, and follow a result back to related measurements, references, and PDG explanatory text. Many lookups return `Option` because the database cannot guarantee that an arbitrary query has a matching record.

The current interface is a working foundation for broader database coverage and more interactive exploration, including a possible terminal UI for organizing related values, footnotes, and references.
