import os
import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import uuid


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def run_from_isolated_script_root():
    if os.environ.get("SIMPLEAI_BASE_SMOKE_CHILD") == "1":
        return False

    runner_dir = tempfile.mkdtemp(prefix="simpleai_base_root_")
    runner_path = os.path.join(runner_dir, "smoke_runner.py")
    try:
        shutil.copyfile(__file__, runner_path)
        env = os.environ.copy()
        env["SIMPLEAI_BASE_SMOKE_CHILD"] = "1"
        result = subprocess.run([sys.executable, "-s", runner_path], cwd=runner_dir, env=env, timeout=180)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return True
    finally:
        shutil.rmtree(runner_dir, ignore_errors=True)


def main():
    from simpleai_base import simpleai_base

    if os.environ.get("SIMPLEAI_BASE_VERIFY_PREFS") == "1":
        expected = json.load(sys.stdin)
        token = simpleai_base.init_local()
        assert_true(token.get_local_mode_vars("user_presets", "") == expected["presets"], "local preferences must survive another process")
        assert_true(token.set_local_mode_vars("user_presets", "B,A") is True, "cross-process preference writes must report success")
        return

    if os.environ.get("SIMPLEAI_BASE_VERIFY_SESSION") == "1":
        expected = json.load(sys.stdin)
        token = simpleai_base.init_local()
        result = json.loads(token.resolve_sstoken(expected["session"], expected["ua"]))
        assert_true(result["status"] == "valid", "session must remain valid in another process")
        assert_true(result["did"] == expected["did"], "session identity must survive another process")
        return

    print("SimpAI base local-mode smoke test ...")
    userhome = tempfile.mkdtemp(prefix="simpleai_base_local_")
    run_id = uuid.uuid4().hex[:10]
    admin_name = f"LocalAdmin_{run_id}"
    admin_phrase = f"Admin123_{run_id}"
    member_name = f"MemberOne_{run_id}"
    member_phrase = f"Member123_{run_id}"
    try:
        token = simpleai_base.init_local()
        token.set_user_base_dir(userhome)

        ua = hashlib.sha256(b"local-browser").hexdigest()
        guest_session = token.get_guest_sstoken(ua)
        assert_true(token.set_local_vars("user_presets", "A,B", guest_session, ua) is True, "valid guest preferences must save")
        assert_true(token.get_local_mode_vars("user_presets", "") == "A,B", "local-mode API must reuse existing guest preferences")
        assert_true(token.set_local_vars("user_presets", "wrong", "expired-session", ua) is False, "invalid sessions must not write Unknown preferences")
        assert_true(token.get_local_vars("user_presets", "rejected", "expired-session", ua) == "rejected", "invalid sessions must not read Unknown preferences")
        assert_true(token.set_local_mode_vars("user_presets", "B") is True, "local-mode saves must not require a browser token")
        assert_true(token.get_local_vars("user_presets", "", guest_session, ua) == "B", "local saves must stay in the guest namespace")
        assert_true(token.set_local_mode_vars("admin_guest_can_generate", "true") is False, "local-mode API must not write admin settings")
        assert_true(token.set_local_vars_for_guest("user_presets", "wrong", guest_session, ua) is False, "guest sync remains admin-only even before an admin exists")
        env = dict(os.environ, SIMPLEAI_BASE_VERIFY_PREFS="1")
        child = subprocess.run(
            [sys.executable, "-s", os.path.abspath(__file__)],
            input=json.dumps({"presets": "B"}), text=True, env=env, timeout=120,
        )
        assert_true(child.returncode == 0, "cross-process preference verification should succeed")
        assert_true(token.get_local_mode_vars("user_presets", "") == "B,A", "cross-process writes must be visible to the original process")

        assert_true(token.get_upstream_did() == "", "upstream DID should be empty in local mode")
        assert_true(token.get_p2p_status() == "Off", "P2P should be disabled in local mode")
        assert_true(token.get_default_workspace_did() == token.get_local_did(), "empty node should use Local workspace DID")

        local_outputs = token.get_path_in_user_dir(token.get_local_did(), "outputs")
        guest_outputs_before_admin = token.get_path_in_user_dir(token.get_guest_did(), "outputs")
        assert_true(os.path.normpath(local_outputs).split(os.sep)[-2] == "Local", "Local DID should use Local folder")
        assert_true(os.path.normpath(guest_outputs_before_admin).split(os.sep)[-2] == "Local", "guest should map to Local before Admin exists")

        os.makedirs(local_outputs, exist_ok=True)
        marker = os.path.join(local_outputs, "local_marker.txt")
        with open(marker, "w", encoding="utf-8") as f:
            f.write("local")

        admin_context = token.set_phrase_and_get_context(admin_name, "", admin_phrase)
        admin_did = admin_context.get_did()
        assert_true(admin_did and not token.is_guest(admin_did), "first local identity should become Admin")
        assert_true(token.get_admin_did() == admin_did, "Admin DID should be set")
        assert_true(token.set_local_mode_vars("user_presets", "wrong") is False, "local-mode writes must be disabled after admin creation")
        try:
            token.get_local_mode_vars("user_presets", "")
        except PermissionError:
            pass
        else:
            raise AssertionError("local-mode reads must be disabled after admin creation")

        admin_outputs = token.get_path_in_user_dir(admin_did, "outputs")
        assert_true(f"{os.sep}admin_" in os.path.normpath(admin_outputs), "Admin should use admin_ folder")
        assert_true(os.path.exists(os.path.join(admin_outputs, "local_marker.txt")), "Local data should migrate into Admin folder")

        guest_outputs_after_admin = token.get_path_in_user_dir(token.get_guest_did(), "outputs")
        assert_true(os.path.normpath(guest_outputs_after_admin).split(os.sep)[-2] == "guest_user", "guest should use guest_user after Admin exists")
        assert_true(not token.can_user_generate(token.get_guest_did()), "guest generation should be disabled by default after Admin exists")
        assert_true(not token.can_user_download_models(token.get_guest_did()), "guest model downloads should be disabled by default after Admin exists")

        token.check_local_user_token(member_name, "")
        member_context = token.set_phrase_and_get_context(member_name, "", member_phrase)
        member_did = member_context.get_did()
        assert_true(member_did and not token.is_guest(member_did), "member identity should bind locally")
        assert_true(not token.can_user_generate(member_did), "pending member should not generate")
        assert_true(not token.can_user_download_models(member_did), "pending member should not download models")
        assert_true(token.approve_user_with_permissions(member_did, True, False) == "OK", "Admin approval should succeed")
        assert_true(token.can_user_generate(member_did), "approved member should generate")
        assert_true(not token.can_user_download_models(member_did), "approved member should not download models by default")
        assert_true(token.set_user_can_download_models(member_did, True) == "OK", "Admin should update member model download permission")
        assert_true(token.can_user_download_models(member_did), "member should download models after permission is enabled")

        old_ua = hashlib.sha256(b"Chrome/140").hexdigest()
        new_ua = hashlib.sha256(b"Chrome/141").hexdigest()
        session = token.get_user_sstoken(admin_did, old_ua)
        member_session = token.get_user_sstoken(member_did, old_ua)
        assert_true(token.set_local_vars("user_presets", "admin-list", session, old_ua) is True, "admin preferences must save")
        assert_true(token.set_local_vars("user_presets", "member-list", member_session, old_ua) is True, "member preferences must save")
        assert_true(token.get_local_vars("user_presets", "", session, new_ua) == "admin-list", "browser upgrades must retain admin preferences")
        assert_true(token.get_local_vars("user_presets", "", member_session, new_ua) == "member-list", "member preferences must be isolated")
        assert_true(token.get_local_vars("user_presets", "", guest_session, ua) == "B,A", "admin creation must preserve guest preferences")
        assert_true(token.set_local_vars("admin_guest_can_generate", "true", member_session, new_ua) is False, "members must not write admin preferences")
        assert_true(token.set_local_vars_for_guest("user_presets", "wrong", member_session, new_ua) is False, "members must not sync guest preferences")
        assert_true(token.set_local_vars_for_guest("user_presets", "guest-default", session, new_ua) is True, "admin guest sync must report success")
        assert_true(token.get_local_vars("user_presets", "", guest_session, ua) == "guest-default", "admin sync must use the original guest namespace")
        assert_true(session.startswith("s2_"), "signed-in browsers should receive persistent random credentials")
        result = json.loads(token.resolve_sstoken(session, new_ua))
        assert_true(result["status"] == "valid" and result["did"] == admin_did, "browser upgrades must preserve identity")
        assert_true(0 < result["expires_in"] <= 90 * 86400, "session idle lifetime must be bounded")
        assert_true(token.check_sstoken_and_get_did(session, new_ua) == admin_did, "existing identity APIs must accept persistent sessions")
        env = os.environ.copy()
        env["SIMPLEAI_BASE_VERIFY_SESSION"] = "1"
        child = subprocess.run(
            [sys.executable, "-s", os.path.abspath(__file__)],
            input=json.dumps({"session": session, "ua": new_ua, "did": admin_did}),
            text=True, env=env, timeout=120,
        )
        assert_true(child.returncode == 0, "cross-process browser session verification should succeed")
        assert_true(token.revoke_sstoken(session), "explicit logout must revoke its credential")
        assert_true(token.set_local_vars("user_presets", "wrong", session, new_ua) is False, "revoked sessions must not save preferences")
        assert_true(json.loads(token.resolve_sstoken(session, old_ua))["status"] == "revoked", "revoked credentials must not renew")
        assert_true(token.check_sstoken_and_get_did(session, old_ua) == "Unknown", "revoked credentials must not grant identity")

        legacy = token.get_legacy_sstoken(admin_did, old_ua)
        migrated = json.loads(token.resolve_sstoken(legacy, old_ua))
        assert_true(migrated["status"] == "valid" and migrated["sstoken"].startswith("s2_"), "valid legacy login should migrate")
        repeated = json.loads(token.resolve_sstoken(legacy, old_ua))
        assert_true(repeated["sstoken"] == migrated["sstoken"], "tabs must share the same legacy upgrade")
        assert_true(token.check_sstoken_and_get_did(legacy, new_ua) == "Unknown", "unmigrated legacy checks must retain their original binding")
        assert_true(token.revoke_sstoken(migrated["sstoken"]), "migrated login should be revocable")
        assert_true(token.check_sstoken_and_get_did(legacy, old_ua) == "Unknown", "legacy credentials must not bypass migrated-session logout")
        assert_true(json.loads(token.resolve_sstoken(legacy, old_ua))["status"] == "revoked", "legacy logout must not issue another upgraded token")

        legacy_ua = hashlib.sha256(b"legacy-second-browser").hexdigest()
        legacy = token.get_legacy_sstoken(admin_did, legacy_ua)
        migrated = json.loads(token.resolve_sstoken(legacy, legacy_ua))
        assert_true(migrated["status"] == "valid", "another browser should migrate independently")
        assert_true(token.revoke_sstoken(legacy), "a stale browser must be able to revoke its legacy credential")
        assert_true(json.loads(token.resolve_sstoken(migrated["sstoken"], new_ua))["status"] == "revoked", "legacy logout must also revoke its upgraded credential")
        assert_true(token.check_sstoken_and_get_did(legacy, legacy_ua) == "Unknown", "revoked legacy credentials must remain rejected")

        print("SimpAI base local-mode and browser-session smoke tests OK")
    finally:
        shutil.rmtree(userhome, ignore_errors=True)


if __name__ == "__main__":
    if not run_from_isolated_script_root():
        main()
