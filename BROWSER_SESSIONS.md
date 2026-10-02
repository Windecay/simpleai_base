# Browser Sessions (0.3.54)

Authenticated browsers receive an opaque, random `s2_` credential. Its encrypted
record is stored in the existing `user_sessions` database tree under a SHA-256
index scoped to the installation DID. Browser User-Agent versions are not part
of the new credential's validation.

- Idle lifetime: 90 days, refreshed on valid use at most once per day.
- Absolute lifetime: 365 days from the original sign-in.
- Explicit unbinding revokes the browser credential.
- The signed user context is checked on every backend validation.
- Revoked, expired, tampered, or foreign-installation credentials never renew.
- Storage failures are distinguished from missing session records.
- Active legacy credentials migrate when `resolve_sstoken()` is called.
- Legacy upgrades reuse the same new credential across tabs; revocation of the
  new credential also rejects the migrated legacy credential.
- Revocation through a legacy credential also revokes its upgraded credential.
- Legacy credentials still require their original User-Agent and time window
  before migration. Already invalid credentials require identity verification.
- No identity files, user passphrases, or environment-derived keys are changed.

Python APIs:

```python
result = json.loads(token.resolve_sstoken(browser_credential, ua_hash))
# status: valid, expired, revoked, invalid, invalid_legacy, or unavailable
# did, sstoken, expires_in are populated only for valid sessions.
token.revoke_sstoken(browser_credential)
```

The Studio integration renews its existing cookie/localStorage representation
only after successful validation. Failed validation uses guest permissions
without overwriting the original browser credential. HttpOnly transport is not
introduced in this release; deployments should use HTTPS where applicable.

The same Rust sources are used for CPython 3.13 (`local-user-mode`) and CPython
3.12 (`build/py312`). Wheels must be built and import-tested for both ABIs.

GitHub Actions runs Rust tests and installed-wheel smoke tests on Linux,
Windows, and macOS. The smoke test covers browser User-Agent changes,
validation from another process, legacy upgrades, and revocation in both
directions. Each wheel artifact includes `SHA256SUMS`; only wheel files are
included in the existing tag-triggered PyPI upload.

`cargo test --lib --locked` uses PyO3's normal Python library linking.
Maturin enables `pyo3/extension-module` through `pyproject.toml` only for wheels;
enabling it unconditionally would prevent Unix Rust test executables from linking.

## Local-mode Preference Persistence (0.3.55)

- Keep the existing token directory, `token.db`, `global_local_vars` tree, and
  `{guest_did}_{sys_did}_{key}` namespace. No migration to a new settings file is
  needed. Existing guest navbar selections remain readable; old `Unknown`
  records are neither imported nor deleted.
- Add `get_local_mode_vars(key, default)` and `set_local_mode_vars(key, value)`
  for server-side local preferences without a browser credential. Both reject
  `admin_` keys and are disabled once an administrator exists. Rejected reads
  raise `PermissionError`; rejected or failed writes return `False`.
- `set_local_vars` and `set_local_vars_for_guest` now return a boolean. Invalid
  or empty identities cannot write preferences, and invalid session reads
  return the caller's default instead of reading an `Unknown` namespace.
  Guest synchronization remains administrator-only.
- Preference writes use `insert_durable`: report success only after insertion
  and Sled flush succeed. The internal `db_insert` request accepts an optional
  `durable` flag so writes forwarded to another base process also request a
  flush. I/O failures are logged without exposing credential or setting values.
- Existing callers can continue ignoring the boolean; new callers should check
  it and read back the value before reporting a successful save. Older wheels
  returning `None` need readback verification in Studio.
- Studio integration uses the local API for navbar preferences and validates
  multi-user identity before saving. Reads and temporary preset/model filtering
  no longer write defaults over a saved selection. The browser only clears its
  draft after the matching successful save acknowledgment; failure or timeout
  retains the draft. The launcher keeps compatible newer installed wheels rather
  than downgrading a manually installed 0.3.55 to 0.3.54.

Validation and release checks:

- `cargo test --lib --locked --offline`: 22 passed, including preference writes,
  dropped-tree write failure, database reopen, and existing browser-session
  tests. The first local attempt lacked Perl for vendored OpenSSL; the completed
  run reused this project's cached OpenSSL headers/libraries via process-local
  `OPENSSL_NO_VENDOR`, `OPENSSL_STATIC`, and `OPENSSL_DIR`. No dependencies were
  downloaded or installed. Existing compiler warnings remain.
- Extend the existing installed-wheel `test.py` smoke test with real guest/local
  namespace continuity, invalid/revoked sessions, cross-process preference
  reads/writes, administrator creation, member isolation, admin-variable denial,
  and administrator-only guest synchronization. Existing CI jobs already run
  this script after installing the newly built wheel.
- No wheel was built or installed locally for this change, and the expanded
  installed-wheel smoke test has not been run locally. GitHub Actions wheel
  builds/smoke tests, any CPython 3.12 branch backport, and live Studio/browser
  acceptance remain release checks. Restart base processes together after
  installation so the internal service also supports durable writes.

## Python 3.12 Branch Synchronization (0.3.55)

- The eight shared preference-persistence files, version/lockfile, installed-wheel
  smoke tests, and this development record are now synchronized to `build/py312`.
  This completes the source-backport item above; both working trees use 0.3.55.
- Keep the branch-specific Python >=3.12 package metadata, Python 3.12
  classifiers, `python3.12` maturin interpreter, `build/py312` CI trigger, and
  `wheels-py312-*` artifact names. The existing three-platform Rust and
  installed-wheel smoke-test steps are unchanged.
- Verified identical diffs for the eight shared files, parsed the Python 3.12
  syntax using the available Python 3.13.13 interpreter, checked Cargo version
  consistency and all three CI platform configurations, and passed
  `git diff --check`. No registered Python 3.12 interpreter was available.
- CPython 3.12 wheel compilation, installed-wheel smoke tests, and live Studio
  acceptance still require the CI artifacts. No staging, commit, push, wheel
  installation, or CI dispatch was performed for this synchronization.
