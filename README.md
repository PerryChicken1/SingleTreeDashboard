# SingleTree dashboard

This repository is a release snapshot of the SingleTree Streamlit dashboard.
It contains application code, runtime assets, pinned Python dependencies, and
a deployment smoke check. Two example inventories (Basel and Evo) are bundled
under `data/examples`, with fixed coordinate systems and manual attribute mapping. Evo's
water bodies and skid roads are bundled too. Other research datasets, notebooks, experiment outputs,
virtual environments, licences, and the development repository's history are
not included. Visitors select Basel for future crop tree selection or Evo for
thinning from below. Custom data uploads, EPSG entry and thinning from above are
not available in this public version. Every attribute dropdown starts at Not set.

## Run locally

Use Python 3.14:

```sh
python -m venv .venv
# Activate .venv using the command for your shell, then:
python -m pip install -r requirements.txt
python smoke_test.py
python -m streamlit run Dashboard.py
```

CBC is the default and is included through OR-Tools. The Gurobi Python package
is also included, but no licence credentials are bundled. Select Linear
Programming, then Gurobi, and enter your WLS Access ID, Secret key and Licence ID
under the runtime limit. The licence must support this cloud host and Gurobi 13.
Desktop `grbgetkey` activation codes cannot be used in these fields.

Credentials are sent to the hosting server and held in the user's session.
They are never saved to files, put in shared caches or exported with results.
Gurobi logging is disabled before credentials are applied, environments are
isolated between runs/users, and models/environments are closed after use.
Use **Clear Gurobi credentials** to erase the fields; changing solver also
clears them. Each user supplies their own credentials; no shared licence is
configured in Streamlit secrets or environment variables.

The automated tests mock WLS authentication: a real user-supplied licence is
still needed to verify authentication from the actual hosting environment.

## Deploy on Streamlit Community Cloud

Connect this GitHub repository and select:

- Branch: `production`
- Main file: `Dashboard.py` (case-sensitive)
- Advanced settings / Python: `3.14`
- App access: public

No owner-supplied secrets or separate system packages are required by the default CBC setup.
Python wheels provide the geospatial dependencies. The GitHub workflow checks
installation, startup, geospatial file loading and optimisation on Linux.
Wait for that check to pass before the first deployment.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Release workflow

`production` is the published version. Push release candidates to a separate
branch, such as `release/2026-09-08`, and open a pull request into `production`.
Inspect the diff, wait for `smoke-test` to pass, then merge only when ready to
update the website. A push to an unrelated branch does not update production.
Do not enable automatic merging for release pull requests.

Protect `production` with required pull requests and the `smoke-test` status
check where your GitHub plan supports it. For a solo maintainer, an additional
person's approving review is not needed. Streamlit itself does not wait for
GitHub checks: the branch protection is what prevents an unchecked release.

Keep regular development in the original source repository. Export a fresh
snapshot deliberately when preparing a release. `release-manifest.json` records
the source HEAD and exported file hashes; snapshots may include uncommitted
source changes, so commit source work first for clearer traceability.

For rollback, revert the release merge using a new pull request into
`production`. Merging the revert redeploys the previous code. Keep dependency
changes in the same release as the code that needs them.

Community Cloud has finite memory/CPU and sleeps after inactivity. This is a
public demo deployment, not a guarantee of capacity for large simultaneous
optimisation runs. Data selections and results are session-based; use Download
results for a CSV. This snapshot does not offer permanent user storage.
# Case-study scope

The Evo case study contains stand_050 (301 trees), lake Särkijärvi and synthetic
skid road 9. The lake and road retain their full saved geometries. Water data:
© OpenStreetMap contributors (ODbL). The two location maps use public-domain
[Natural Earth country boundaries](https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-admin-0-countries-2/)
with markers derived from the case-study tree coordinates.
