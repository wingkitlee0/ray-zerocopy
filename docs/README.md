# Documentation Build

This directory contains the Sphinx documentation for ray-zerocopy.

## Building Documentation

### Install dependencies

```bash
pip install -r requirements.txt
```

Or install from pyproject.toml:

```bash
pip install -e ".[docs]"
```

### Build HTML documentation

```bash
cd docs
sphinx-build -b html . _build/html
```

View the documentation by opening `_build/html/index.html` in your browser.

### Clean build

```bash
rm -rf _build
sphinx-build -b html . _build/html
```

## Documentation Structure

```
docs/
├── index.md                    # Landing page
├── getting_started.md          # Installation and quick start
├── user_guide/                 # User guides
│   ├── index.md
│   ├── core_concepts.md
│   ├── tasks.md
│   ├── actors.md
│   ├── torchscript.md
│   └── ray_data_integration.md
├── tutorials/                  # Step-by-step tutorials
│   ├── index.md
│   ├── basic_inference.md
│   ├── pipeline_example.md
│   └── ray_data_batch.md
├── api_reference/              # API documentation
│   ├── index.md
│   ├── wrappers.md
│   └── model_wrappers.md
├── migration.md                # Migration guide
├── conf.py                     # Sphinx configuration
└── requirements.txt            # Documentation dependencies
```

## ReadTheDocs

This project is configured to build on ReadTheDocs using `.readthedocs.yaml` in the project root.

## Writing Documentation

- Use Markdown (.md files) via MyST-Parser
- Use Google-style docstrings in Python code
- Reference code with {py:class}, {py:func}, etc.
- Cross-reference with `[text](path.md)`
