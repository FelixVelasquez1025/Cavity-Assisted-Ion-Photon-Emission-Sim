# Contributing

## Introduction

Welcome to CAIPE-sim! This is only the first draft of the contribution guidelines, which are expected to evolve as we progress and discovery good processes.

## Getting Started

Here is an overview of the contribution process:

1. Clone the repository to your local machine.
  - Make sure you use the correct method, depending on whether you want HTTPS or SSH connections.
  - Run `uv sync` to initialize the project and activate the virtual environment.
2. Create a new branch for your changes.
  - Branch names should be descriptive and follow the format `feature/feature-name` or `bugfix/bug-description`. If linked to an issue, use the format `iss-number`, e.g `iss-05`.
3. Make your changes and commit them locally.
  - Include short but descriptive commit messages.
  - Use the imperative mood, e.g. "add feature" instead of "added feature".
4. Push your changes to the remote (publish the branch if you have to).
5. Create a pull request to the main repository from your branch.
  - If you need to build on code from a PR that has not yet been approved, just create a branch from the existing PR branch.

See the following sections for more details.

## Coding Standards

To save ourselves some headaches later, we should begin by following good practices for coding style, unit/integration testing, and documentation.

### Coding Style

We'll try using the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html).

You can [set Black as your default formatter in VSCode](https://code.visualstudio.com/docs/python/formatting).

### Unit Testing

We'll use the [`unittest`](https://docs.python.org/3/library/unittest.html) module as our testing framework. Tests live in the `/tests` directory. Unit tests should run before every PR. Some quick AI-drafted tips:

- **Keep Tests Small and Focused**
  - Each test should verify a single, well-defined behavior or function.
  - Avoid testing multiple aspects in one test to simplify debugging and maintenance.
- **Test Independence**
  - Ensure tests do not depend on each other or share state.
  - Use fixtures to set up and tear down test environments, keeping tests isolated.
- **Readable and Descriptive Tests**
  - Use clear, descriptive names for test functions and classes (e.g., `test_addition_with_negative_numbers`).
  - Add comments or docstrings to clarify complex test logic.
- **One Assertion per Test (When Possible)**
  - Aim for a single assertion per test to pinpoint failures easily, though multiple related assertions can be grouped if necessary.
- **Test Edge Cases and Error Conditions**
  - Cover typical, boundary, and error scenarios to ensure robustness.
- **Use Proper Naming Conventions**
  - Prefix test files with `test_` and test methods with `test_` for automatic discovery by test runners.
- **Automate and Run Tests Regularly**
  - Integrate tests into continuous integration (CI) pipelines to catch regressions early.
- **Maintain and Refactor Tests**
  - Update tests as code evolves and refactor them to keep them relevant and effective.
- **Monitor Code Coverage**
  - Use tools like `coverage.py` to ensure critical code paths are tested, aiming for high coverage but prioritizing meaningful tests over metrics.

### Documentation

Follow the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).

Write documentation as the last step in a PR so that you don't have to rewrite it as you fix bugs, etc.

## Pull Requests

All changes should be submitted through pull requests for review and documentation purposes. Where possible, pull requests should be [linked to a GitHub issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).
