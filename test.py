import os
import shutil
import tempfile

from simpleai_base import simpleai_base


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    print("SimpAI base local-mode smoke test ...")
    userhome = tempfile.mkdtemp(prefix="simpleai_base_local_")
    try:
        token = simpleai_base.init_local()
        token.set_user_base_dir(userhome)

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

        admin_context = token.set_phrase_and_get_context("LocalAdmin", "", "Admin123")
        admin_did = admin_context.get_did()
        assert_true(admin_did and not token.is_guest(admin_did), "first local identity should become Admin")
        assert_true(token.get_admin_did() == admin_did, "Admin DID should be set")

        admin_outputs = token.get_path_in_user_dir(admin_did, "outputs")
        assert_true(f"{os.sep}admin_" in os.path.normpath(admin_outputs), "Admin should use admin_ folder")
        assert_true(os.path.exists(os.path.join(admin_outputs, "local_marker.txt")), "Local data should migrate into Admin folder")

        guest_outputs_after_admin = token.get_path_in_user_dir(token.get_guest_did(), "outputs")
        assert_true(os.path.normpath(guest_outputs_after_admin).split(os.sep)[-2] == "guest_user", "guest should use guest_user after Admin exists")
        assert_true(not token.can_user_generate(token.get_guest_did()), "guest generation should be disabled by default after Admin exists")
        assert_true(not token.can_user_download_models(token.get_guest_did()), "guest model downloads should be disabled by default after Admin exists")

        token.check_local_user_token("MemberOne", "")
        member_context = token.set_phrase_and_get_context("MemberOne", "", "Member123")
        member_did = member_context.get_did()
        assert_true(member_did and not token.is_guest(member_did), "member identity should bind locally")
        assert_true(not token.can_user_generate(member_did), "pending member should not generate")
        assert_true(not token.can_user_download_models(member_did), "pending member should not download models")
        assert_true(token.approve_user_with_permissions(member_did, True, False) == "OK", "Admin approval should succeed")
        assert_true(token.can_user_generate(member_did), "approved member should generate")
        assert_true(not token.can_user_download_models(member_did), "approved member should not download models by default")
        assert_true(token.set_user_can_download_models(member_did, True) == "OK", "Admin should update member model download permission")
        assert_true(token.can_user_download_models(member_did), "member should download models after permission is enabled")

        print("SimpAI base local-mode smoke test OK")
    finally:
        shutil.rmtree(userhome, ignore_errors=True)


if __name__ == "__main__":
    main()
