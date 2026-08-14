+++
title = "splotrs"
+++
{{ project_header(project="splotrs") }}

[`splotrs`](https://github.com/denehoffman/splotrs) performs unbinned mixture fits and calculates a per-event sWeight for each component. Its numerical core is written in Rust using `ganesh`, while its Python interface accepts vectorized NumPy probability-density callbacks.

The fit estimates component yields together with shared PDF shape parameters. Results include yield and shape uncertainties, the event-summed sPlot yield covariance, the complete joint fit covariance, convergence diagnostics, and sWeights in the same row order as the input data.

## Two interfaces

Python users provide a two-dimensional event array and one normalized PDF callback per component. Optional shape parameters, initial yields, signed event weights, and optimizer controls customize the fit:

```python
result = splot(data, pdfs, shape_parameters=parameters)
print(result.yields)
print(result.sweights)
```

Rust users can implement `ParametricPdf` or supply compatible closures and call the native API without crossing the Python boundary. The project is distributed both as a Python package and as a Rust crate.

## Statistical contract

Component PDFs must be normalized over the discriminating variables and sufficiently distinct to produce an invertible covariance matrix. Variables studied with the resulting sWeights should be independent of those discriminating variables within each component. Signed input weights are supported, although some weighted samples can still lead to a singular or indefinite information matrix.
