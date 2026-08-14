+++
title = "yamloom"
+++
{{ project_header(project="yamloom") }}

[`yamloom`](https://github.com/denehoffman/yamloom) generates GitHub Actions workflows from Python objects. Its goal is to make workflow authoring feel like normal typed programming: constructors expose the allowed workflow keys, reusable actions provide focused interfaces, and Python functions and loops replace repeated YAML.

The generated workflows can be checked against SchemaStore's GitHub Actions schema before they are written. Yamloom also combines job-level permissions with the permissions recommended by individual actions, retaining the access each step needs without requiring one large hand-maintained permissions block.

## A workflow as code

Workflows are assembled from `Workflow`, `Job`, event, action, and script objects. Expressions expose GitHub contexts and support operations for conditions, matrix values, and other fields that GitHub evaluates at runtime.

Yamloom's synchronization command treats a Python generator as the source of truth for a complete set of workflow files:

```console
yamloom
yamloom check
yamloom convert .github/workflows/checks.yml -o imported_workflow.py
```

Generated files carry ownership metadata, so synchronization can update or remove its own stale outputs without touching manual workflows. Existing YAML can be converted into a structurally equivalent generator, with known actions represented by typed shortcuts and unknown actions preserved through a general fallback.

The public interface is Python, while the serialization core is implemented in Rust and exposed through PyO3. The package can be installed with `pip install yamloom` or `uv pip install yamloom`.
