# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ripple is a JAX-based package for differentiable and hardware-accelerated gravitational wave (GW) data analysis. It implements multiple waveform approximants that match lalsuite implementations to machine precision across the full parameter space. The package is designed for performance with float32 precision by default but can be configured for float64.

## Development Commands

### Testing
```bash
# Run all tests (excluding old_tests)
uv run pytest --cov=ripplegw --cov-report=term-missing tests/ --ignore=tests/old_tests

# Run a single test file
uv run pytest tests/test_conversion.py

# Run a specific test function
uv run pytest tests/test_conversion.py::test_conversion
```

### Linting and Formatting
```bash
# Run ruff linter (with auto-fix)
uv run ruff check --fix src/

# Run ruff formatter
uv run ruff format src/

# Run pre-commit hooks on all files
uv run pre-commit run --all-files
```

### Type Checking
Note: Pyright is currently disabled in pre-commit due to existing type errors. Re-enable once type errors are fixed.

### Documentation
```bash
# Build documentation locally
uv run mkdocs serve
```

### Installation
```bash
# Install dependencies (including dev dependencies)
uv sync --all-extras --dev

# Install with CUDA support
uv sync --extra cuda --dev
```

## Code Architecture

### Waveform Module Structure

All waveform implementations are in `src/ripplegw/waveforms/`. The package follows a modular design where:

- **Main waveform files** (e.g., `IMRPhenomXAS.py`, `IMRPhenomD.py`, `IMRPhenomXPHM.py`, `TaylorF2.py`) contain the primary generation functions
- **LALSim* files** contain ported functions from lalsuite, maintaining close correspondence to the original LAL implementations
- **Utility files** (e.g., `IMRPhenomD_utils.py`, `IMRPhenomX_utils.py`) provide shared calculations and coefficients

### Parameter Convention

Waveform functions follow a consistent parameter array structure. For example, `gen_IMRPhenomXAS_hphc(f, params, f_ref)` expects:

```python
params = [Mc, eta, chi1, chi2, D, tc, phic, inclination]
```

Where:
- `Mc`: Chirp mass in solar masses
- `eta`: Symmetric mass ratio (0.0 to 0.25)
- `chi1`, `chi2`: Dimensionless aligned spins (-1 to 1)
- `D`: Luminosity distance in Mpc
- `tc`: Time of coalescence (seconds)
- `phic`: Phase at coalescence (radians)
- `inclination`: Inclination angle (0 to π)

Convert between component masses and chirp mass/eta using:
```python
from ripplegw import ms_to_Mc_eta, Mc_eta_to_ms
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
```

### Core Utilities

The main `__init__.py` (`src/ripplegw/__init__.py`) provides fundamental utilities:
- Mass conversions: `ms_to_Mc_eta`, `Mc_eta_to_ms`
- Tidal parameter conversions: `lambdas_to_lambda_tildes`, `lambda_tildes_to_lambdas`
- Waveform comparison: `get_match`, `get_match_arr`, `get_phase_maximized_inner_product`
- Constants are defined in `src/ripplegw/constants.py`

### Waveform Families

**Implemented waveforms:**
- `IMRPhenomXAS`: Aligned spin, extensively tested
- `IMRPhenomXPHM`: Precessing, higher modes (actively being validated)
- `IMRPhenomD`: Aligned spin
- `IMRPhenomPv2`: Precessing (finalizing sampling validation)
- `TaylorF2`: With tidal effects
- `IMRPhenomD_NRTidalv2`: Tidal effects (verified for low spin χ₁,χ₂ < 0.05)

Each waveform typically provides:
- A main generation function (e.g., `gen_IMRPhenomXAS_hphc` returns both h+ and hx polarizations)
- Internal phase/amplitude calculation functions
- Frequency-dependent coefficient computations

### JAX-Specific Considerations

- **JIT Compilation**: Always JIT-compile waveform functions for performance. Avoid recompilation by keeping frequency array lengths constant.
- **Float Precision**: Default is float32. Enable float64 with:
  ```python
  from jax import config
  config.update("jax_enable_x64", True)
  ```
- **No Index Errors**: JAX does not raise index errors, so ensure parameter arrays are correctly ordered
- **Immutability**: JAX arrays are immutable; operations return new arrays

### Testing Structure

- `tests/` contains current test files
- `tests/old_tests/` contains deprecated tests that should be ignored
- Test files often use `lalsimulation` and `bilby` to validate waveforms against reference implementations
- Benchmarking scripts (e.g., `benchmark_waveform.py`) test both accuracy (mismatch) and speed

### Branch Information

- Main branch for PRs: `main`
- Current branch: `xphm_timing` (working on timing/performance for XPHM waveforms)

## Important Notes

- The package uses `uv` for dependency management rather than pip or poetry
- Pre-commit hooks include ruff linting and formatting
- Coverage reports are uploaded to Coveralls via CI
- Documentation is built with mkdocs-material and includes Jupyter notebook support
