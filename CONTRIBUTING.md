# Contributing to Duck Egg Fertility Detection

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional, for containerized development)

### Setting Up Development Environment

1. **Fork the repository**

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/duck_egg_fertility_detection.git
   cd duck_egg_fertility_detection
   ```

3. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

   # Using conda
   conda create -n duck_egg python=3.10
   conda activate duck_egg
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-web.txt
   ```

5. **Set up pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

6. **Create a .env file**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following our [Coding Standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

Run the test suite before committing:

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_ML/      # ML tests
pytest tests/test_web/     # Web API tests

# Run with coverage
pytest --cov=src/ --cov-report=html
```

### 4. Run Linting

```bash
# Check code formatting
black --check src/ tests/ web/

# Run linter
ruff check src/ tests/ web/

# Type checking
mypy src/
```

### 5. Commit Changes

Follow conventional commits:

```bash
git add .
git commit -m "feat: add new feature X"
git commit -m "fix: resolve issue Y"
git commit -m "docs: update README"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request through GitHub.

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all functions
- Write docstrings for all public functions/classes

### Code Example

```python
from typing import Optional
import numpy as np

def preprocess_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Preprocess an input image for prediction.
    
    Args:
        image: Input image as numpy array
        threshold: Binarization threshold value
        
    Returns:
        Preprocessed image as numpy array
    """
    if image is None:
        raise ValueError("Input image cannot be None")
    
    # Processing logic here
    processed = image * threshold
    return processed
```

### Naming Conventions

- **Variables**: `snake_case` (e.g., `image_path`, `batch_size`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Classes**: `PascalCase` (e.g., `ImagePreprocessor`)
- **Functions**: `snake_case` (e.g., `load_model()`)

### Git Commit Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what and why
3. **Screenshots**: For UI changes
4. **Checklist**:
   - [ ] Tests pass
   - [ ] Linting passes
   - [ ] Documentation updated
   - [ ] Related issues linked

### Code Review Process

1. Maintainers will review your PR
2. Address feedback if requested
3. Once approved, maintainers will merge your PR

## Reporting Issues

### Bug Reports

Include:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- System information

### Feature Requests

Include:
- Clear description of the feature
- Use cases
- Potential implementation suggestions

---

## Questions?

Open an issue for discussion or contact the maintainers.

Thank you for contributing! 
