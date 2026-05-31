# Contributing to GMM_clustering

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/GMM_clustering.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Set up development environment:
   ```bash
   conda create -n gmm-dev python=3.9
   conda activate gmm-dev
   pip install -r requirements.txt
   ```

## Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Write docstrings for all functions (NumPy style)
- Keep line length ≤ 88 characters

## Testing

Before submitting a PR:

```bash
pytest tests/
flake8 src/
black --check src/
```

Add tests for new features in `tests/`.

## Pull Request Process

1. Update documentation and docstrings
2. Add/update tests
3. Run all tests locally
4. Update CHANGELOG.md
5. Submit PR with clear description of changes
6. Address review comments

## Reporting Issues

Use GitHub Issues to report bugs or suggest features. Include:
- Clear description
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and environment info

## Questions?

Feel free to open an issue for questions or discussions.
