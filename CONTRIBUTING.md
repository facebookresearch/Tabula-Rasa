# Contributing to Tabula Rasa

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

This repository is the public home of the project. Development happens here, and
changes are reviewed as pull requests against `main`.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

### Running the tests

For the Python renderer:

```bash
cd python
pip install -e .
python tests/test_two_pass_independence.py
```

`tests/` holds numerical validation scripts rather than unit tests, so run them
directly with `python`. `test_two_pass_independence.py` is quick;
`test_bias_variance.py` decomposes MSE into bias and variance and takes
considerably longer.

For the WebGPU demo, serve `webgpu/` over HTTP and open the pages under
`webgpu/tests/` in a WebGPU-capable browser. Each sets the tab title to `PASS`
or `FAIL`.

Note that `webgpu/tests/mesh-orientation-test.html` reads the source files as
text and asserts on the `Params` uniform layout, which is declared in three WGSL
files and packed by hand in `main.js`. A mismatch between those four places is
silent at runtime, so if you change `Params`, change it in all four and keep that
test passing.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

- **Python**: 4 spaces, `snake_case` for functions and variables, `CapWords` for
  classes. Imports are relative within the `tabula_rasa` package and absolute
  outside it. Avoid `import *`.
- **JavaScript**: 4 spaces, `camelCase`. No build step or bundler; the demo is
  plain ES2020 loaded directly by the browser.
- **WGSL**: 4 spaces. Shared declarations belong in `shaders/common.wgsl`.

## Third-party code

If you add code derived from another project, add an entry to
`THIRD_PARTY_NOTICES.md` and keep any required attribution comment in the source
file itself. Do not remove existing attribution blocks; several are license
conditions rather than courtesies.

## License

By contributing to Tabula Rasa, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
