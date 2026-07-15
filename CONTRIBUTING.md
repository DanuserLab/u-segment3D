# Contributing to u-Segment3D

Thank you for your interest in contributing to u-Segment3D. Contributions can include bug reports, feature requests, documentation improvements, tutorials, tests, examples, and code changes.

Please follow the project [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## Reporting Issues

Before opening a new issue, please check whether a similar issue already exists. When reporting a bug, include:

- A short description of the problem and expected behavior.
- The u-Segment3D version or commit used.
- Operating system, Python version, and installation method.
- Relevant package versions, especially Cellpose, PyTorch/CUDA, CuPy, or MATLAB if applicable.
- A minimal example, command, script, screenshot, or error log that helps reproduce the issue.
- Whether the issue occurs with the Python package, MATLAB interface, or both.

Please do not upload large or private microscopy datasets directly to GitHub. Use a small reproducible example when possible, or describe how maintainers can access appropriate test data.

## Development Setup

A typical local development setup is:

```bash
git clone https://github.com/DanuserLab/u-segment3D.git
cd u-segment3D
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

GPU-enabled workflows depend on the user's local CUDA, PyTorch, Cellpose, and CuPy environment. If a contribution affects GPU-dependent behavior, please document the tested hardware and software versions.

## Pull Requests

For code or documentation changes:

- Create a focused branch for the change.
- Keep pull requests small and specific when possible.
- Describe what changed, why it changed, and how it was tested.
- Preserve existing Python and MATLAB entry points unless a breaking change is discussed first.
- Update documentation, tutorials, or parameter descriptions when user-facing behavior changes.
- Avoid committing generated files, large datasets, local paths, credentials, or machine-specific configuration.

## Testing and Validation

Before submitting a pull request, run the most relevant checks for the change. At minimum, verify that the package imports:

```bash
python -c "import segment3D; print(segment3D.__file__)"
```

For changes affecting workflows, run a small example dataset or tutorial script when possible. For changes affecting the MATLAB package interface, run the MATLAB test script in `software/testuSegment3DMatlabPack.m` or the relevant workflow step.

If a full test cannot be run because it requires large data, GPU hardware, MATLAB, or HPC resources, explain what was tested and what remains untested in the pull request.

## Coding and Documentation Guidelines

- Follow the style and structure of the surrounding code.
- Prefer clear, maintainable implementations over large rewrites.
- Keep algorithmic behavior stable unless the change is intentional and documented.
- Add comments only where they clarify non-obvious logic.
- Use small example data or synthetic data for tests and tutorials whenever possible.
- Document new parameters, expected inputs, outputs, and limitations.

## License and Citation

u-Segment3D is distributed under the GNU General Public License v3.0. By contributing, you agree that your contribution will be distributed under the same license.

If you use u-Segment3D in scientific work, please cite the associated publication listed in the README.
