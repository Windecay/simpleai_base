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
