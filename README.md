# causal_ai

Playground for causal inference methods in data science. This repository contains tutorials and examples for learning and experimenting with causal inference using Python libraries like DoWhy.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) for package management

#### Installing uv

On Windows with PowerShell:
```powershell
iwr https://astral.sh/uv/install.ps1 -useb | iex
```

On macOS or Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip:
```bash
pip install uv
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jpardovega/causal_ai.git
   cd causal_ai
   ```

2. Install dependencies with uv:
   ```bash
   uv pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tutorials

The tutorials are located in the `tutorials/` directory. You can run them directly with Python:

```bash
python tutorials/datacamp_tutorial.py
```

Or open them in VS Code and use the interactive cells (marked with `# %%`).

## Development

- The project uses `ruff` for linting and formatting
- Pre-commit hooks are configured to ensure code quality
- Run all checks manually with:
  ```bash
  pre-commit run --all-files
  ```
