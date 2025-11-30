# Publishing and Versioning Guide

This document describes how to version and publish `ray-zerocopy` to PyPI.

## Version Management

The project uses git tags for version management. Versions are automatically determined from git tags using `setuptools-scm` during the build process.

### Version Format

Versions follow [Semantic Versioning](https://semver.org/) with a `v` prefix:
- `v1.0.0` - Major release
- `v1.1.0` - Minor release
- `v1.1.1` - Patch release
- `v0.1.0` - Pre-1.0 release

### Creating a New Release

1. **Update version tag:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Automated publishing:**
   - The GitHub Actions workflow (`.github/workflows/publish.yml`) automatically triggers on tag push
   - It builds the package using `uv build`
   - Publishes to PyPI using trusted publishing (or API token)

3. **Verify release:**
   - Check PyPI: https://pypi.org/project/ray-zerocopy/
   - Verify installation: `pip install ray-zerocopy==1.0.0`

## PyPI Setup

### Option 1: Trusted Publishing (Recommended)

1. Go to https://pypi.org/manage/account/
2. Navigate to "API tokens" → "Add API token"
3. Create a token for the project (scope: `ray-zerocopy`)
4. In GitHub repository settings:
   - Go to Settings → Secrets and variables → Actions
   - The workflow uses trusted publishing, so no token is needed if your GitHub account is linked to PyPI

### Option 2: API Token (Alternative)

If not using trusted publishing:

1. Create a PyPI API token at https://pypi.org/manage/account/token/
2. Add it as a GitHub Secret:
   - Repository → Settings → Secrets and variables → Actions
   - Add secret: `PYPI_API_TOKEN` with your token value
3. Update `.github/workflows/publish.yml` to use the token:
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

## Local Testing

Before publishing, test the build locally:

```bash
# Install build dependencies
uv pip install uv_build setuptools-scm

# Build the package
uv build

# Check the built package
ls -la dist/

# Test install (optional)
pip install dist/ray_zerocopy-*.whl
```

## ReadTheDocs Setup

1. Go to https://readthedocs.org/
2. Import your GitHub repository: `wingkitlee0/ray-zerocopy`
3. Configure:
   - Python version: 3.11
   - Configuration file: `.readthedocs.yaml` (already configured)
4. Enable "Build pull requests" if desired
5. Documentation will auto-build on pushes to main/master

The `.readthedocs.yaml` file is already configured to:
- Use Python 3.11
- Build Sphinx documentation from `docs/`
- Install the package and documentation dependencies

## Version in Code

The version is automatically available in Python:

```python
from ray_zerocopy import __version__
print(__version__)  # e.g., "1.0.0"
```

The version is determined from git tags at build time and written to `src/ray_zerocopy/_version.py` by `setuptools-scm`.

## Troubleshooting

### Version not updating
- Ensure git tags are pushed: `git push --tags`
- Check that `setuptools-scm` is in `[build-system]` requires
- Verify tag format matches `v*` pattern

### Build fails
- Ensure you have a git tag: `git tag -l`
- Check that the repository has git history: `git log --oneline`

### PyPI publish fails
- Verify PyPI credentials are set up correctly
- Check GitHub Actions logs for specific errors
- Ensure the package name is available on PyPI (not taken by another project)
