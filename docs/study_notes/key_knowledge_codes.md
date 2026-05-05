# Difference: `venv` vs `conda`

This is very important.

## 1. `venv` (Python built-in)

Comes with Python itself.

Characteristics:
- Only manages Python packages
- Uses `pip`
- Lightweight
- Simple

Example:

```bash
python -m venv myenv
```

## 2. Conda Environment

Comes with Anaconda / Miniconda.

Characteristics:
- Manages Python version
- Manages packages
- Manages system-level dependencies (C libraries, CUDA, etc.)
- Uses `conda` + `pip`
- Better for data science / ML

## Direct Comparison

| Feature | `venv` | `conda` |
|---|---|---|
| Python version control | Limited | Full |
| Package manager | `pip` only | `conda` + `pip` |
| Handles C/CUDA dependencies | No | Yes |
| Best for | Web development | AI / ML / Data science |
| GPU support setup | Harder | Easier |

## Simple Analogy

- `venv` = small toolbox
- `conda` = full workshop
