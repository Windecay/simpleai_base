use base58::{FromBase58, ToBase58};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::dids::token_utils;

pub(crate) const PREFIX: &str = "s2_";
pub(crate) const IDLE_TIMEOUT: u64 = 90 * 24 * 3600;
pub(crate) const ABSOLUTE_TIMEOUT: u64 = 365 * 24 * 3600;
const TOUCH_INTERVAL: u64 = 24 * 3600;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BrowserSession {
    pub did: String,
    pub sys_did: String,
    context_hash: String,
    created_at: u64,
    last_seen_at: u64,
    expires_at: u64,
    pub revoked: bool,
}

impl BrowserSession {
    pub fn new(did: &str, sys_did: &str, context_sig: &str, now: u64) -> Self {
        Self {
            did: did.to_owned(),
            sys_did: sys_did.to_owned(),
            context_hash: URL_SAFE_NO_PAD.encode(token_utils::calc_sha256(context_sig.as_bytes())),
            created_at: now,
            last_seen_at: now,
            expires_at: now.saturating_add(IDLE_TIMEOUT),
            revoked: false,
        }
    }

    pub fn status(&self, sys_did: &str, now: u64) -> &'static str {
        if self.sys_did != sys_did
            || self.created_at > now.saturating_add(300)
            || self.last_seen_at < self.created_at
            || self.last_seen_at > now.saturating_add(300)
            || self.expires_at > self.created_at.saturating_add(ABSOLUTE_TIMEOUT)
            || self.expires_at > self.last_seen_at.saturating_add(IDLE_TIMEOUT)
        {
            "invalid"
        } else if self.revoked {
            "revoked"
        } else if now >= self.expires_at
            || now >= self.created_at.saturating_add(ABSOLUTE_TIMEOUT)
        {
            "expired"
        } else {
            "valid"
        }
    }

    pub fn matches_context(&self, context_sig: &str) -> bool {
        self.context_hash
            == URL_SAFE_NO_PAD.encode(token_utils::calc_sha256(context_sig.as_bytes()))
    }

    pub fn touch(&mut self, now: u64) -> bool {
        if self.status(&self.sys_did, now) != "valid"
            || now < self.last_seen_at.saturating_add(TOUCH_INTERVAL)
        {
            return false;
        }
        self.last_seen_at = now;
        self.expires_at = now
            .saturating_add(IDLE_TIMEOUT)
            .min(self.created_at.saturating_add(ABSOLUTE_TIMEOUT));
        true
    }

    pub fn expires_in(&self, now: u64) -> u64 {
        self.expires_at.saturating_sub(now)
    }

    pub fn encode(&self, key: &[u8; 32]) -> Result<String, serde_json::Error> {
        let bytes = serde_json::to_vec(self)?;
        Ok(URL_SAFE_NO_PAD.encode(token_utils::encrypt(&bytes, key, 0)))
    }

    pub fn decode(encoded: &str, key: &[u8; 32]) -> Option<Self> {
        let encrypted = URL_SAFE_NO_PAD.decode(encoded).ok()?;
        // The existing decrypt helper assumes a complete AES-GCM nonce.
        if encrypted.len() < 28 {
            return None;
        }
        serde_json::from_slice(&token_utils::decrypt(&encrypted, key, 0)).ok()
    }
}

pub(crate) fn new_token() -> String {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    format!("{}{}", PREFIX, bytes.to_base58())
}

pub(crate) fn storage_key(token: &str, sys_did: &str) -> Option<String> {
    let bytes = token.strip_prefix(PREFIX)?.from_base58().ok()?;
    if bytes.len() != 32 {
        return None;
    }
    Some(format!(
        "browser_v2:{}:{}",
        sys_did,
        URL_SAFE_NO_PAD.encode(token_utils::calc_sha256(token.as_bytes()))
    ))
}

pub(crate) fn encode_upgrade(token: &str, key: &[u8; 32]) -> String {
    URL_SAFE_NO_PAD.encode(token_utils::encrypt(token.as_bytes(), key, 0))
}

pub(crate) fn decode_upgrade(encoded: &str, key: &[u8; 32]) -> Option<String> {
    let encrypted = URL_SAFE_NO_PAD.decode(encoded).ok()?;
    if encrypted.len() < 28 {
        return None;
    }
    let token = String::from_utf8(token_utils::decrypt(&encrypted, key, 0)).ok()?;
    storage_key(&token, "")?;
    Some(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: u64 = 1_700_000_000;

    fn session() -> BrowserSession {
        BrowserSession::new("user", "system", "signed-context", START)
    }

    #[test]
    fn tokens_are_random_and_only_hashes_are_used_as_storage_keys() {
        let first = new_token();
        let second = new_token();
        assert_ne!(first, second);
        let key = storage_key(&first, "system").unwrap();
        assert!(!key.contains(&first));
        assert_ne!(Some(key), storage_key(&first, "other-system"));
    }

    #[test]
    fn malformed_tokens_do_not_resolve() {
        for token in ["", "s2_", "s2_0", "s2_111", "legacy"] {
            assert!(storage_key(token, "system").is_none());
        }
    }

    #[test]
    fn idle_expiry_is_not_a_global_time_bucket() {
        let record = session();
        assert_eq!(record.status("system", START + 46 * 24 * 3600), "valid");
        assert_eq!(record.status("system", START + IDLE_TIMEOUT - 1), "valid");
        assert_eq!(record.status("system", START + IDLE_TIMEOUT), "expired");
    }

    #[test]
    fn active_sessions_renew_without_extending_the_absolute_deadline() {
        let mut record = session();
        for day in 1..365 {
            assert!(record.touch(START + day * 24 * 3600));
            assert_eq!(record.status("system", START + day * 24 * 3600), "valid");
        }
        assert_eq!(record.expires_in(START + 364 * 24 * 3600), 24 * 3600);
        assert!(!record.touch(START + ABSOLUTE_TIMEOUT));
        assert_eq!(record.status("system", START + ABSOLUTE_TIMEOUT), "expired");
    }

    #[test]
    fn expired_or_revoked_sessions_cannot_be_renewed() {
        let mut record = session();
        assert!(!record.touch(START + IDLE_TIMEOUT));
        record.revoked = true;
        assert_eq!(record.status("system", START + 1), "revoked");
        assert!(!record.touch(START + TOUCH_INTERVAL));
    }

    #[test]
    fn touches_are_throttled_and_do_not_move_time_backwards() {
        let mut record = session();
        assert!(!record.touch(START));
        assert!(!record.touch(START - 1));
        assert!(!record.touch(START + TOUCH_INTERVAL - 1));
        assert!(record.touch(START + TOUCH_INTERVAL));
        assert!(!record.touch(START + TOUCH_INTERVAL));
    }

    #[test]
    fn browser_versions_are_not_part_of_persistent_session_validation() {
        let record = session();
        assert_eq!(record.status("system", START + 1), "valid");
        assert!(record.matches_context("signed-context"));
        assert!(!record.matches_context("new-signed-context"));
        assert_eq!(record.status("other-system", START + 1), "invalid");
    }

    #[test]
    fn future_or_inconsistent_timestamps_are_rejected() {
        let mut record = session();
        assert_eq!(record.status("system", START - 301), "invalid");
        record.expires_at = START + ABSOLUTE_TIMEOUT + 1;
        assert_eq!(record.status("system", START), "invalid");
        record = session();
        record.last_seen_at = START - 1;
        assert_eq!(record.status("system", START), "invalid");
    }

    #[test]
    fn encrypted_records_survive_restart_and_reject_tampering() {
        let key = [7u8; 32];
        let encoded = session().encode(&key).unwrap();
        assert!(!encoded.contains("signed-context"));
        let restored = BrowserSession::decode(&encoded, &key).unwrap();
        assert_eq!(restored.status("system", START + 1), "valid");
        assert!(BrowserSession::decode(&encoded, &[8u8; 32]).is_none());
        let mut bytes = URL_SAFE_NO_PAD.decode(&encoded).unwrap();
        bytes[20] ^= 1;
        assert!(BrowserSession::decode(&URL_SAFE_NO_PAD.encode(bytes), &key).is_none());
    }

    #[test]
    fn truncated_records_do_not_panic() {
        for length in 0..28 {
            let encoded = URL_SAFE_NO_PAD.encode(vec![0u8; length]);
            assert!(BrowserSession::decode(&encoded, &[0u8; 32]).is_none());
        }
        assert!(BrowserSession::decode("invalid!", &[0u8; 32]).is_none());
    }

    #[test]
    fn legacy_upgrade_targets_are_encrypted_and_validate_token_format() {
        let token = new_token();
        let key = [23u8; 32];
        let encoded = encode_upgrade(&token, &key);
        assert!(!encoded.contains(&token));
        assert_eq!(decode_upgrade(&encoded, &key), Some(token));
        assert!(decode_upgrade(&encoded, &[24u8; 32]).is_none());
        assert!(decode_upgrade(&encode_upgrade("Unknown", &key), &key).is_none());
        assert!(decode_upgrade("AA", &key).is_none());
    }
}
