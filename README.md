# COMET

The Constant pressure COntroller using METropolis algorithm (COMET) command-line interface.

## Quickstart

Install the package in editable mode:

```bash
pip install -e .
```

Run the example workflow configuration from the repository root:

```bash
comet run examples/config.yaml
```

You will need to provide your own:

- MACE model directory
- LAMMPS trajectory file
- H₂ template file
- Region/box definition file

Show the available commands:

```bash
comet --help
```

The `comet` module can also be invoked directly:

```bash
python -m comet --help
```

## Configuration

An example configuration is provided at `examples/config.yaml`. The loader keeps keys and
structure unchanged using `yaml.safe_load`.
