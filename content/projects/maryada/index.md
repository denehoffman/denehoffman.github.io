+++
title = "maryada"
+++
<!-- markdownlint-disable MD034 -->
{{ project_header(title="maryada", summary="A no_std binary64 interval arithmetic library conforming to IEEE Std 1788.1-2017.", tags=["Rust", "Interval arithmetic", "no_std", "IEEE 1788.1"], repository="https://github.com/denehoffman/maryada", documentation="https://docs.rs/maryada", package="https://crates.io/crates/maryada") }}

[`maryada`](https://github.com/denehoffman/maryada) is a `no_std` binary64 interval arithmetic library conforming to IEEE Std 1788.1-2017. It provides bare and decorated real intervals, outward-rounded elementary operations, and text and binary interchange. Rectangular complex intervals are available through an optional feature.

The conformance claim applies to the real interval API. The rectangular complex interval extension is outside the scope of the standard. The repository's [conformance statement](https://github.com/denehoffman/maryada/blob/main/CONFORMANCE.md) documents the operation accuracy declarations, required features, implementation details, and test coverage.

## A small interval

```rust
use maryada::Interval;

let x = Interval::new(1.0, 2.0);
let y = x.sqr();

assert_eq!(y.bounds(), (1.0, 4.0));
```

The optional `complex` feature enables rectangular complex intervals through `ComplexBox`; `num-complex` adds interoperability with `num_complex::Complex64`. Complex functions transform interval spaces in nontrivial ways, so their results contain the true image but are not guaranteed to equal it under every operation.

Future directions include other complex interval formulations such as disks and polyarcs, linear algebra methods, and potentially Python bindings.
