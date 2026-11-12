# Releasing

crypto-quant follows [Semantic Versioning](https://semver.org/). While the version is `0.x`, the public API may change
between minor versions.

| Change | Bump | Example |
|---|---|---|
| Incompatible change to the CLI, config keys or public functions | MAJOR (MINOR while in `0.x`) | removing a config key |
| New feature, model, metric or notable result change | MINOR | adding a new model |
| Bug fix or documentation-only change | PATCH | fixing a metric calculation |

## Steps

1. Make sure `main` is green (CI and docs workflows pass).
2. Open a release PR (e.g. branch `release/vX.Y.Z`) that:
   - renames the `[Unreleased]` section of `CHANGELOG.md` to `[X.Y.Z] - YYYY-MM-DD`, adds a new empty `[Unreleased]`
     section above it, and updates the comparison links at the bottom;
   - sets `version = "X.Y.Z"` in `pyproject.toml`.
3. Get the PR reviewed and merge it.
4. A maintainer tags the merge commit on `main` and pushes the tag:
   ```bash
   git checkout main && git pull --ff-only
   git tag -a vX.Y.Z -m "crypto-quant vX.Y.Z"
   git push origin vX.Y.Z
   ```
5. The [Release workflow](.github/workflows/release.yml) then checks that the tag matches the package version,
   builds the sdist and wheel, and publishes a GitHub Release with the notes from `CHANGELOG.md`.
6. Check the release page and the attached files.

If the workflow fails, fix the problem in a new PR, delete the tag (`git push --delete origin vX.Y.Z` and
`git tag -d vX.Y.Z`) and tag again.
