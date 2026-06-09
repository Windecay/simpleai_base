use std::collections::HashMap;
use std::env::Args;
use std::fs;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use base58::{FromBase58, ToBase58};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use bytes::Bytes;
use chrono::format;
use prometheus_client::metrics::info;
use qrcode::render::svg;
use qrcode::{EcLevel, QrCode, Version};
use serde::de;
use serde_json::{self, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::field::debug;
use tracing_subscriber::EnvFilter;

use pyo3::prelude::*;

use crate::api;
use crate::dids::cert_center::GlobalCerts;
use crate::dids::claims::{GlobalClaims, IdClaim, UserContext};
use crate::dids::TOKEN_ENTRYPOINT_DID;
use crate::dids::{self, token_utils, tokendb::TokenDB, DidToken};
use crate::p2p::{
    self, DidMessage, P2pRequest, P2pServer, DEFAULT_P2P_CONFIG, P2P_HANDLE, P2P_INSTANCE,
};
use crate::user::shared::{self, SharedData};
use crate::user::user_mgr::{MessageQueue, OnlineUsers};
use crate::user::user_vars::{AdminDefault, GlobalLocalVars};
use crate::user::{DidEntryPoint, TokenUser};
use crate::utils::env_data::EnvData;
use crate::utils::error::TokenError;
use crate::utils::systeminfo::SystemInfo;
use crate::{exchange_key, issue_key};

pub(crate) static TOKEN_API_VERSION: &str = "v1.2.2";

static SYNC_TASK_HANDLE: Mutex<Option<tokio::task::JoinHandle<()>>> = Mutex::new(None);

#[derive(Clone)]
#[pyclass]
pub struct SimpleAI {
    pub sys_name: String,
    pub sys_did: String,
    pub node_id: String,
    pub device_did: String,
    pub guest_did: String,

    didtoken: Arc<Mutex<DidToken>>,
    tokenuser: Arc<Mutex<TokenUser>>,
    token_db: Arc<RwLock<TokenDB>>, //HashMap<String, serde_json::Value>,
    global_local_vars: Arc<RwLock<GlobalLocalVars>>, //HashMap<global|admin|{did}_{key}, String>,
    online_users: OnlineUsers,
    last_timestamp: Arc<RwLock<u64>>,
    sid_did_map: Arc<Mutex<HashMap<String, String>>>,
    shared_data: &'static SharedData,
    p2p_config: String,
    p2p_status: Option<api::P2pStatus>,
}

#[pymethods]
impl SimpleAI {
    #[new]
    pub fn new() -> Self {
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init();
        //println!("This is test version for some case, not release version!");
        let (system_name, sys_phrase, device_name, device_phrase, guest_name, guest_phrase) =
            dids::get_system_vars();
        debug!(
            "system_name:{}, device_name:{}, guest_name:{}",
            system_name, device_name, guest_name
        );

        let didtoken = DidToken::instance();
        let global_local_vars = GlobalLocalVars::instance();
        let tokenuser = TokenUser::instance();

        let (sys_did, device_did, guest_did, token_db) = {
            let didtoken = didtoken.lock().unwrap();
            (
                didtoken.get_sys_did(),
                didtoken.get_device_did(),
                didtoken.get_guest_did(),
                didtoken.get_token_db(),
            )
        };

        let online_users = OnlineUsers::new(60, 2);
        let message_queue = MessageQueue::new(global_local_vars.clone());
        let mut shared_data = shared::get_shared_data();
        shared_data.set_message_queue(message_queue);
        shared_data.set_sys_data(&sys_did, &device_did, &system_name);

        let admin_did = didtoken.lock().unwrap().get_admin_did();
        if !admin_did.is_empty() {
            //online_users.log_register(admin_did.clone());
            shared_data.online_all.log_register(admin_did.clone());
        }

        Self {
            sys_name: system_name,
            sys_did,
            node_id: "".to_string(),
            device_did,
            guest_did,
            didtoken,
            tokenuser,
            token_db,
            global_local_vars,
            online_users,
            last_timestamp: Arc::new(RwLock::new(0u64)),
            sid_did_map: Arc::new(Mutex::new(HashMap::new())),
            shared_data,
            p2p_config: DEFAULT_P2P_CONFIG.to_string(),
            p2p_status: None,
        }
    }

    pub fn get_sys_name(&self) -> String {
        self.sys_name.clone()
    }
    pub fn get_sys_did(&self) -> String {
        self.sys_did.clone()
    }
    pub fn get_node_id(&self) -> String {
        self.node_id.clone()
    }
    pub fn set_node_id(&mut self, node_id: &str) {
        self.node_id = node_id.to_string();
        self.didtoken.lock().unwrap().set_node_id(node_id);
    }
    pub fn get_device_did(&self) -> String {
        self.device_did.clone()
    }
    pub fn get_guest_did(&self) -> String {
        self.guest_did.clone()
    }
    pub fn is_guest(&self, did: &str) -> bool {
        did == self.guest_did.as_str()
    }
    pub fn get_local_did(&self) -> String {
        self.get_sys_did()
    }
    pub fn is_local_user(&self, did: &str) -> bool {
        did == self.get_local_did() || (self.absent_admin() && self.is_guest(did))
    }
    pub fn get_default_workspace_did(&self) -> String {
        if self.absent_admin() {
            self.get_local_did()
        } else {
            self.get_guest_did()
        }
    }

    pub fn get_sysinfo(&self) -> SystemInfo {
        self.didtoken.lock().unwrap().get_sysinfo()
    }

    pub fn get_node_mode(&mut self) -> String {
        let system_did = self.get_sys_did();
        let mut node_mode = self.global_local_vars.read().unwrap().get_local_vars(
            "node_mode_type",
            "local",
            &system_did,
        );
        if node_mode == "online" {
            node_mode = "local".to_string();
            self.global_local_vars.write().unwrap().set_local_vars(
                "node_mode_type",
                &node_mode,
                &system_did,
            );
        }
        {
            let mut didtoken = self.didtoken.lock().unwrap();
            didtoken.set_node_mode(&node_mode);
            didtoken.get_node_mode()
        }
    }

    pub fn set_node_mode(&mut self, mode: &str) {
        let mode = if mode == "online" { "local" } else { mode };
        let system_did = self.get_sys_did();
        let current_mode = self.global_local_vars.read().unwrap().get_local_vars(
            "node_mode_type",
            "local",
            &system_did,
        );
        if mode != current_mode.as_str() {
            self.global_local_vars.write().unwrap().set_local_vars(
                "node_mode_type",
                mode,
                &self.get_sys_did(),
            );
            self.didtoken.lock().unwrap().set_node_mode(&mode);
        }
    }

    pub(crate) fn get_admin_did(&self) -> String {
        self.didtoken.lock().unwrap().get_admin_did()
    }

    pub(crate) fn set_admin_did(&mut self, did: &str) {
        if !did.is_empty() {
            self.log_register(&did);
        }
        self.didtoken.lock().unwrap().set_admin_did(did);
        self.global_local_vars.write().unwrap().set_admin_did(did);
    }

    pub fn is_admin(&self, did: &str) -> bool {
        did == self.get_admin_did()
    }

    pub fn absent_admin(&self) -> bool {
        self.get_admin_did().is_empty()
    }

    pub fn get_p2p_upstream_did(&mut self) -> String {
        String::new()
    }

    pub fn disconnect_upstream(&mut self) {
        self.didtoken.lock().unwrap().set_upstream_did("");
        self.p2p_stop();
        let mut handle_guard = SYNC_TASK_HANDLE.lock().unwrap();
        if let Some(handle) = handle_guard.take() {
            handle.abort();
            *handle_guard = None;
        }
    }

    pub fn get_upstream_did(&mut self) -> String {
        self.didtoken.lock().unwrap().set_upstream_did("");
        String::new()
    }

    pub(crate) fn p2p_start(&mut self) -> String {
        self.set_node_id("");
        self.p2p_status = Some(api::P2pStatus::default());
        "P2P disabled in local mode".to_string()
    }

    /// 停止 P2P 服务
    pub(crate) fn p2p_stop(&mut self) -> String {
        self.set_node_id("");
        self.p2p_status = Some(api::P2pStatus::default());
        "P2P disabled in local mode".to_string()
    }

    /// 重启 P2P 服务
    pub(crate) fn p2p_restart(&mut self) -> String {
        self.p2p_stop()
    }

    pub fn get_p2p_status(&mut self) -> String {
        self.p2p_status = Some(api::P2pStatus::default());
        "Off".to_string()
    }
    pub fn get_p2p_is_debug(&mut self) -> bool {
        if self.p2p_status.is_none() {
            self.get_p2p_status();
        }
        self.p2p_status
            .as_ref()
            .map_or(false, |status| status.is_debug)
    }

    pub fn get_p2p_is_running(&mut self) -> bool {
        if self.p2p_status.is_none() {
            if self.get_p2p_status() == "Off" {
                return false;
            }
        }
        self.p2p_status
            .as_ref()
            .map_or(false, |status| status.node_id != "")
    }

    pub fn get_p2p_address(&mut self) -> String {
        if !self.get_p2p_is_running() {
            return "".to_string();
        }
        let p2p_node_did = self
            .p2p_status
            .as_ref()
            .map_or("".to_string(), |status| status.node_did.clone());
        let short_sys_did = self.get_sys_did().chars().take(7).collect::<String>();
        let p2p_address = format!("{}.{}", short_sys_did, p2p_node_did);
        p2p_address
    }

    pub fn request_remote_task(
        &mut self,
        task_id: &str,
        task_method: &str,
        args: Vec<u8>,
        target_did: Option<String>,
        mode: Option<String>,
    ) -> String {
        "".to_string()
    }

    pub fn response_remote_task(
        &mut self,
        task_id: &str,
        task_method: &str,
        result: Vec<u8>,
    ) -> String {
        "".to_string()
    }

    pub fn get_global_status(&self, sid: &str, last_timestamp: u64) -> (usize, usize, usize) {
        let last_time = self.last_timestamp.read().unwrap();
        let user_list = self.online_users.get_full_list();
        let did = self
            .sid_did_map
            .lock()
            .unwrap()
            .get(sid)
            .cloned()
            .unwrap_or_default();
        self.shared_data
            .get_last(&did, last_timestamp, Some(&user_list))
    }

    pub fn get_online_users_number(&self) -> usize {
        self.online_users.get_number()
    }

    pub fn get_online_nodes_users(&self) -> (usize, usize) {
        self.online_users.get_nodes_users()
    }

    pub fn get_online_nodes_top(&self) -> String {
        self.online_users.get_nodes_top_list()
    }

    pub fn log_register(&self, sid: &str) {
        let did = self
            .sid_did_map
            .lock()
            .unwrap()
            .get(sid)
            .cloned()
            .unwrap_or_default();
        self.online_users.log_register(did.to_string());
        self.shared_data.online_all.log_register(did.to_string());
    }

    pub fn log_access(&mut self, sid: &str) -> (usize, usize, usize, usize) {
        let did = self
            .sid_did_map
            .lock()
            .unwrap()
            .get(sid)
            .cloned()
            .unwrap_or_default();
        self.online_users.log_access(did.to_string());
        self.shared_data.online_all.log_access(did.to_string());
        let (domain_online_nodes, domain_online_users) = if self.get_p2p_is_running() {
            self.online_users.get_nodes_users()
        } else {
            (0, 0)
        };
        (
            self.online_users.get_number(),
            domain_online_nodes,
            domain_online_users,
            self.shared_data
                .get_message_queue()
                .get_msg_number(&self.get_sys_did()),
        )
    }

    pub fn get_global_msg_number(&self) -> usize {
        self.shared_data
            .get_message_queue()
            .get_msg_number(&self.get_sys_did())
    }

    pub fn get_global_msg_all(&self) -> String {
        self.shared_data
            .get_message_queue()
            .get_messages(&self.get_sys_did(), 0)
    }

    pub fn remove_old_global_msg(&self, timestamp: u64) {
        self.shared_data
            .get_message_queue()
            .remove_old_messages(&self.get_sys_did(), timestamp);
    }

    pub fn get_global_msg_list(&self, last_timestamp: u64) -> String {
        self.shared_data
            .get_message_queue()
            .get_messages(&self.get_sys_did(), last_timestamp)
    }

    pub fn put_global_message(&self, message: &str) {
        let _ = self
            .shared_data
            .get_message_queue()
            .push_messages(&self.get_sys_did(), message.to_string());
    }

    pub fn get_global_vars(&mut self, key: &str, default: &str) -> String {
        self.global_local_vars
            .read()
            .unwrap()
            .get_global_vars(key, default)
    }

    pub fn put_global_var(&mut self, key: &str, value: &str) {
        self.global_local_vars
            .write()
            .unwrap()
            .put_global_var(key, value)
    }

    pub fn get_global_vars_json(&mut self) -> String {
        self.global_local_vars
            .read()
            .unwrap()
            .get_global_vars_json()
    }

    pub fn get_local_vars(
        &mut self,
        key: &str,
        default: &str,
        user_session: &str,
        ua_hash: &str,
    ) -> String {
        let user_did = self.check_sstoken_and_get_did(user_session, ua_hash);
        self.global_local_vars
            .read()
            .unwrap()
            .get_local_vars(key, default, &user_did)
    }

    pub fn get_local_admin_vars(&mut self, key: &str) -> String {
        self.global_local_vars
            .read()
            .unwrap()
            .get_local_admin_vars(key)
    }

    pub fn set_local_vars(&mut self, key: &str, value: &str, user_session: &str, ua_hash: &str) {
        let user_did = self.check_sstoken_and_get_did(user_session, ua_hash);
        self.global_local_vars
            .write()
            .unwrap()
            .set_local_vars(key, value, &user_did)
    }

    pub fn set_local_admin_vars(
        &mut self,
        key: &str,
        value: &str,
        user_session: &str,
        ua_hash: &str,
    ) {
        let user_did = self.check_sstoken_and_get_did(user_session, ua_hash);
        if user_did == self.get_admin_did() {
            self.global_local_vars.write().unwrap().set_local_vars(
                &format!("admin_{}", key),
                value,
                &user_did,
            );
            if key == "p2p_in_did_list" {
                self.shared_data.set_p2p_in_dids(&value);
                let p2p_node_did = self
                    .p2p_status
                    .as_ref()
                    .map_or("".to_string(), |status| status.node_did.clone());

                //println!("{} [SimpBase] {} is in p2p_in_did_list set to: {}", token_utils::now_string(), value, self.shared_data.is_p2p_in_dids(value))
            } else if key == "p2p_out_did_list" {
                self.shared_data.set_p2p_out_dids(&value);
                let p2p_node_did = self
                    .p2p_status
                    .as_ref()
                    .map_or("".to_string(), |status| status.node_did.clone());

                //println!("{} [SimpBase] {} is in p2p_out_did_list set to: {}", token_utils::now_string(), value, self.shared_data.is_p2p_out_dids(value))
            }
        }
    }

    pub fn set_local_vars_for_guest(
        &mut self,
        key: &str,
        value: &str,
        user_session: &str,
        ua_hash: &str,
    ) {
        let user_did = self.check_sstoken_and_get_did(user_session, ua_hash);
        self.global_local_vars
            .write()
            .unwrap()
            .set_local_vars_for_guest(key, value, &user_did)
    }

    pub fn can_user_generate(&self, did: &str) -> bool {
        self.global_local_vars
            .read()
            .unwrap()
            .can_user_generate(did)
    }

    pub fn can_user_download_models(&self, did: &str) -> bool {
        self.global_local_vars
            .read()
            .unwrap()
            .can_user_download_models(did)
    }

    pub fn get_user_access_list(&self) -> String {
        self.global_local_vars
            .read()
            .unwrap()
            .get_user_access_list()
    }

    pub fn approve_user(&mut self, did: &str, can_generate: bool) -> String {
        if !IdClaim::validity(did) {
            return "Unknown".to_string();
        }
        let claim = self.get_claim(did);
        if claim.is_default() {
            return "Unknown".to_string();
        }
        let cert = self.didtoken.lock().unwrap().issue_local_member_cert(did);
        if cert == "Unknown" {
            return "Unknown".to_string();
        }
        self.global_local_vars
            .write()
            .unwrap()
            .approve_user(did, &claim.nickname, can_generate);
        "OK".to_string()
    }

    pub fn approve_user_with_permissions(
        &mut self,
        did: &str,
        can_generate: bool,
        can_download_models: bool,
    ) -> String {
        if !IdClaim::validity(did) {
            return "Unknown".to_string();
        }
        let claim = self.get_claim(did);
        if claim.is_default() {
            return "Unknown".to_string();
        }
        let cert = self.didtoken.lock().unwrap().issue_local_member_cert(did);
        if cert == "Unknown" {
            return "Unknown".to_string();
        }
        self.global_local_vars
            .write()
            .unwrap()
            .approve_user_with_permissions(
                did,
                &claim.nickname,
                can_generate,
                can_download_models,
            );
        "OK".to_string()
    }

    pub fn reject_user(&mut self, did: &str) -> String {
        if !IdClaim::validity(did) {
            return "Unknown".to_string();
        }
        let nickname = self.get_claim(did).nickname;
        self.global_local_vars
            .write()
            .unwrap()
            .reject_user(did, &nickname);
        "OK".to_string()
    }

    pub fn set_user_can_generate(&mut self, did: &str, can_generate: bool) -> String {
        if !IdClaim::validity(did) {
            return "Unknown".to_string();
        }
        self.global_local_vars
            .write()
            .unwrap()
            .set_user_can_generate(did, can_generate);
        "OK".to_string()
    }

    pub fn set_user_can_download_models(&mut self, did: &str, can_download_models: bool) -> String {
        if !IdClaim::validity(did) {
            return "Unknown".to_string();
        }
        self.global_local_vars
            .write()
            .unwrap()
            .set_user_can_download_models(did, can_download_models);
        "OK".to_string()
    }

    pub fn set_guest_can_generate(&mut self, can_generate: bool) -> String {
        self.global_local_vars
            .write()
            .unwrap()
            .set_guest_can_generate(can_generate);
        "OK".to_string()
    }

    pub fn set_guest_can_download_models(&mut self, can_download_models: bool) -> String {
        self.global_local_vars
            .write()
            .unwrap()
            .set_guest_can_download_models(can_download_models);
        "OK".to_string()
    }

    pub fn get_guest_can_generate(&self) -> bool {
        self.global_local_vars
            .read()
            .unwrap()
            .get_guest_can_generate()
    }

    pub fn get_guest_can_download_models(&self) -> bool {
        self.global_local_vars
            .read()
            .unwrap()
            .get_guest_can_download_models()
    }

    pub fn migrate_local_workspace_to_admin(&self, admin_did: &str) -> String {
        self.tokenuser
            .lock()
            .unwrap()
            .migrate_local_workspace_to_admin(admin_did)
    }

    pub(crate) fn get_pending_did_list(&self, way: &str) -> String {
        self.global_local_vars
            .read()
            .unwrap()
            .get_pending_did_list(way)
    }
    pub(crate) fn add_pending_did(&mut self, did: &str, way: &str) {
        self.global_local_vars
            .write()
            .unwrap()
            .add_pending_did(did, way)
    }
    pub(crate) fn remove_pending_did(&mut self, did: &str, way: &str) {
        self.global_local_vars
            .write()
            .unwrap()
            .remove_pending_did(did, way)
    }

    pub(crate) fn get_allowed_did_list(&self, way: &str) -> String {
        self.global_local_vars
            .read()
            .unwrap()
            .get_allowed_did_list(way)
    }
    pub(crate) fn add_allowed_did(&mut self, did: &str, way: &str) {
        self.global_local_vars
            .write()
            .unwrap()
            .add_allowed_did(did, way)
    }
    pub(crate) fn remove_allowed_did(&mut self, did: &str, way: &str) {
        self.global_local_vars
            .write()
            .unwrap()
            .remove_allowed_did(did, way)
    }

    pub fn reset_admin(&mut self, admin_did: &str) -> String {
        if IdClaim::validity(admin_did) {
            let admin_claim = self.get_claim(admin_did);
            let is_registered = self.is_registered(admin_did);

            if !admin_claim.is_default() && is_registered {
                let old_admin = self.get_admin_did();
                self.set_admin_did(admin_did);
                {
                    let mut tokenuser = self.tokenuser.lock().unwrap();
                    tokenuser.remove_context(admin_did);
                    tokenuser.remove_context(&old_admin);
                }
                println!(
                    "{} [SimpBase] reset_admin to {}",
                    token_utils::now_string(),
                    admin_did
                );
                return "OK".to_string();
            }
        }
        "Unknown".to_string()
    }

    pub fn reset_node_mode(&mut self, mode: &str) -> (String, String, String) {
        let node_mode = self.get_node_mode();
        if mode == "isolated" && node_mode != "isolated" {
            println!(
                "{} [SimpBase] reset node mode to isolated",
                token_utils::now_string()
            );
            // 清除非 device，system，guest 的 crypt_secrets
            let remove_dids = self
                .didtoken
                .lock()
                .unwrap()
                .remove_crypt_secrets_for_users();
            // 清除非 guest 的 token
            {
                let mut tokenuser = self.tokenuser.lock().unwrap();
                for did in &remove_dids {
                    tokenuser.remove_context(did);
                }
            }
            let (system_name, sys_phrase, device_name, device_phrase, guest_name, guest_phrase) =
                dids::get_system_vars();
            let admin_name = guest_name.replace("guest_", "admin_");
            let admin_symbol_hash = IdClaim::get_symbol_hash_by_source(
                &admin_name,
                Some("8610000000001".to_string()),
                None,
            );
            let (admin_hash_id, admin_phrase) =
                token_utils::get_key_hash_id_and_phrase("User", &admin_symbol_hash);
            let admin_did = {
                let user_did = self
                    .didtoken
                    .lock()
                    .unwrap()
                    .reverse_lookup_did_by_symbol(admin_symbol_hash);
                let identity_file = token_utils::get_path_in_sys_key_dir(&format!(
                    "user_identity_{}.token",
                    admin_hash_id
                ));
                if user_did != "Unknown" && identity_file.exists() {
                    let encrypted_identity = fs::read_to_string(identity_file.clone())
                        .expect(&format!("Unable to read file: {}", identity_file.display()));
                    self.tokenuser.lock().unwrap().import_user(
                        &URL_SAFE_NO_PAD.encode(admin_symbol_hash),
                        &encrypted_identity,
                        &admin_phrase,
                    );
                    user_did
                } else {
                    let (admin_did, admin_phrase) = self.tokenuser.lock().unwrap().create_user(
                        &admin_name,
                        &String::from("8610000000001"),
                        None,
                        None,
                    );
                    admin_did
                }
            };
            let admin_phrase_base58 = admin_phrase.as_bytes().to_base58();
            println!(
                "{} [SimpBase] local admin/本地管理身份: did/标识={}, phrase/口令={}",
                token_utils::now_string(),
                admin_did,
                admin_phrase_base58
            );
            self.set_admin_did(&admin_did);
            self.set_node_mode(mode);
            self.tokenuser
                .lock()
                .unwrap()
                .sign_user_context(&admin_did, &admin_phrase);
            (admin_did, admin_name, admin_phrase_base58)
        } else if mode == "online" && node_mode != "online" {
            //
            println!(
                "{} [SimpBase] reset node mode to online",
                token_utils::now_string()
            );
            let admin_did = self.get_admin_did();
            if !admin_did.is_empty() {
                let _remove_dids = self
                    .didtoken
                    .lock()
                    .unwrap()
                    .remove_crypt_secrets_for_users();
                self.tokenuser.lock().unwrap().remove_context(&admin_did);
            }
            self.set_admin_did("");
            self.set_node_mode(mode);
            ("".to_string(), "".to_string(), "".to_string())
        } else {
            ("".to_string(), "".to_string(), "".to_string())
        }
    }

    pub fn export_isolated_admin_qrcode_svg(&mut self) -> String {
        if self.get_node_mode() == "isolated" && !self.get_admin_did().is_empty() {
            let admin = self.get_admin_did();
            let admin_claim = self.get_claim(&admin);
            let qrcode_svg = SimpleAI::export_user_qrcode_svg(&admin);
            if !qrcode_svg.is_empty() {
                format!("{}|{}|{}", admin_claim.nickname, admin, qrcode_svg)
            } else {
                "".to_string()
            }
        } else {
            "".to_string()
        }
    }

    #[staticmethod]
    pub fn export_user_qrcode_svg(user_did: &str) -> String {
        let encrypted_identity_qr_base64 = SimpleAI::export_user_qrcode_base64(user_did);
        if !encrypted_identity_qr_base64.is_empty() {
            let qrcode = QrCode::with_version(
                encrypted_identity_qr_base64,
                Version::Normal(12),
                EcLevel::L,
            )
            .unwrap();
            let image = qrcode
                .render()
                .min_dimensions(400, 400)
                .dark_color(svg::Color("#800000"))
                .light_color(svg::Color("#ffff80"))
                .build();
            image
        } else {
            "".to_string()
        }
    }

    #[staticmethod]
    pub(crate) fn export_user_qrcode_base64(user_did: &str) -> String {
        let didtoken = DidToken::instance();
        let claim = didtoken.lock().unwrap().get_claim(user_did);
        if !claim.is_default() {
            let user_symbol_hash = claim.get_symbol_hash();
            let (user_hash_id, _user_phrase) =
                token_utils::get_key_hash_id_and_phrase("User", &user_symbol_hash);
            let identity_file = token_utils::get_path_in_sys_key_dir(&format!(
                "user_identity_{}.token",
                user_hash_id
            ));
            match identity_file.exists() {
                true => {
                    let identity = fs::read_to_string(identity_file.clone())
                        .expect(&format!("Unable to read file: {}", identity_file.display()));
                    let encrypted_identity = URL_SAFE_NO_PAD.decode(identity.clone()).unwrap();
                    let did_bytes = user_did.from_base58().unwrap();
                    let user_cert = {
                        let certificates = GlobalCerts::instance();
                        let user_cert = certificates.lock().unwrap().get_register_cert(user_did);
                        user_cert
                    };
                    debug!(
                        "{} [SimpBase] user_cert:{}",
                        token_utils::now_string(),
                        user_cert
                    );
                    let user_cert_bytes = token_utils::get_slim_user_cert(&user_cert);
                    if user_cert_bytes.len() < 120 {
                        return "".to_string();
                    }
                    let mut encrypted_identity_qr = Vec::with_capacity(
                        encrypted_identity.len() + did_bytes.len() + user_cert_bytes.len(),
                    );
                    encrypted_identity_qr.extend_from_slice(&did_bytes);
                    encrypted_identity_qr.extend_from_slice(&user_cert_bytes);
                    encrypted_identity_qr.extend_from_slice(&encrypted_identity);
                    URL_SAFE_NO_PAD.encode(encrypted_identity_qr.clone())
                }
                false => "".to_string(),
            }
        } else {
            "".to_string()
        }
    }

    #[staticmethod]
    pub fn import_identity_qrcode(encrypted_identity: &str) -> (String, String, String) {
        let identity = URL_SAFE_NO_PAD.decode(encrypted_identity).unwrap();
        let (user_did, nickname, telephone, user_cert) =
            token_utils::import_identity_qrcode(&identity);
        if user_did != "Unknown" && user_cert != "Unknown" {
            debug!(
                "import_identity_qrcode, ready to push user cert: did={}",
                user_did
            );
            let certificates = GlobalCerts::instance();
            certificates.lock().unwrap().push_user_cert_text(&format!(
                "{}|{}|{}|{}",
                TOKEN_ENTRYPOINT_DID, user_did, "Member", user_cert
            ));
        }
        (user_did, nickname, telephone)
    }

    pub fn get_entry_point(&self, user_did: &str, entry_point_id: &str) -> String {
        if user_did == self.get_admin_did() {
            token_utils::gen_entry_point_of_service(entry_point_id)
        } else {
            "".to_string()
        }
    }

    pub fn get_guest_sstoken(&mut self, ua_hash: &str) -> String {
        let guest_did = self.get_guest_did();
        self.get_user_sstoken(&guest_did, ua_hash)
    }

    pub fn get_user_sstoken(&mut self, did: &str, ua_hash: &str) -> String {
        if IdClaim::validity(did) {
            let now_sec = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_secs();
            let context = self.tokenuser.lock().unwrap().get_user_context(did);
            if context.is_default() || context.is_expired() {
                println!(
                    "{} [SimpBase] The user context is error or expired: did={}",
                    token_utils::now_string(),
                    did
                );
                return String::from("Unknown");
            }
            let text1 = self.didtoken.lock().unwrap().get_local_crypt_text(ua_hash);
            let text2 = token_utils::calc_sha256(format!("{}", now_sec / 2000000).as_bytes());
            let mut text_bytes: [u8; 64] = [0; 64];
            text_bytes[..32].copy_from_slice(&text1);
            text_bytes[32..].copy_from_slice(&text2);
            let text_hash = token_utils::calc_sha256(&text_bytes);
            let did_bytes = did
                .from_base58()
                .unwrap_or("Unknown".to_string().into_bytes());
            let mut padded_did_bytes: [u8; 32] = [0; 32];
            padded_did_bytes[..11].copy_from_slice(&did_bytes[10..]);
            padded_did_bytes[11..].copy_from_slice(&did_bytes);
            let result: [u8; 32] = text_hash
                .iter()
                .zip(padded_did_bytes.iter())
                .map(|(&a, &b)| a ^ b)
                .collect::<Vec<u8>>()
                .try_into()
                .expect("get_user_sstoken, Failed to convert Vec<u8> to [u8; 32]");
            result.to_base58()
        } else {
            debug!("debug: get_user_sstoken, did is incorrect format: {}", did);
            String::from("Unknown")
        }
    }

    pub fn check_sstoken_and_get_did(&mut self, sstoken: &str, ua_hash: &str) -> String {
        let sstoken_bytes = sstoken.from_base58().unwrap_or([0; 32].to_vec());
        if sstoken_bytes.len() != 32 || sstoken_bytes == [0; 32] {
            println!(
                "{} [SimpBase] The sstoken is incorrect format: {}",
                token_utils::now_string(),
                sstoken
            );
            return String::from("Unknown");
        }
        let mut padded_sstoken_bytes: [u8; 32] = [0; 32];
        padded_sstoken_bytes.copy_from_slice(&sstoken_bytes);
        let now_sec = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();
        let text1 = self.didtoken.lock().unwrap().get_local_crypt_text(ua_hash);
        let text2 = token_utils::calc_sha256(format!("{}", now_sec / 2000000).as_bytes());
        let mut text_bytes: [u8; 64] = [0; 64];
        text_bytes[..32].copy_from_slice(&text1);
        text_bytes[32..].copy_from_slice(&text2);
        let text_hash = token_utils::calc_sha256(&text_bytes);
        let result: [u8; 32] = text_hash
            .iter()
            .zip(padded_sstoken_bytes.iter())
            .map(|(&a, &b)| a ^ b)
            .collect::<Vec<u8>>()
            .try_into()
            .expect("check_sstoken_and_get_did, 1, Failed to convert Vec<u8> to [u8; 32]");
        let mut did_bytes: [u8; 21] = [0; 21];
        let mut padded: [u8; 11] = [0; 11];
        padded.copy_from_slice(&result[..11]);
        did_bytes.copy_from_slice(&result[11..]);
        let did_bytes_slice = &did_bytes[10..];
        if padded
            .iter()
            .zip(did_bytes_slice.iter())
            .all(|(a, b)| a == b)
        {
            let user_did = did_bytes.to_base58();
            let context = self.tokenuser.lock().unwrap().get_user_context(&user_did);
            if context.is_default() || context.is_expired() {
                self.sid_did_map.lock().unwrap().remove(sstoken);
                println!(
                    "{} [SimpBase] The context of the sstoken is expired: did={}",
                    token_utils::now_string(),
                    user_did
                );
                String::from("Unknown")
            } else {
                self.sid_did_map
                    .lock()
                    .unwrap()
                    .insert(sstoken.to_string(), user_did.clone());
                user_did
            }
        } else {
            let text2 = token_utils::calc_sha256(format!("{}", now_sec / 2000000 - 1).as_bytes());
            let mut text_bytes: [u8; 64] = [0; 64];
            text_bytes[..32].copy_from_slice(&text1);
            text_bytes[32..].copy_from_slice(&text2);
            let text_hash = token_utils::calc_sha256(&text_bytes);
            let result: [u8; 32] = text_hash
                .iter()
                .zip(padded_sstoken_bytes.iter())
                .map(|(&a, &b)| a ^ b)
                .collect::<Vec<u8>>()
                .try_into()
                .expect("check_sstoken_and_get_did, 2, Failed to convert Vec<u8> to [u8; 32]");
            padded.copy_from_slice(&result[..11]);
            did_bytes.copy_from_slice(&result[11..]);
            let did_bytes_slice = &did_bytes[10..];
            if padded
                .iter()
                .zip(did_bytes_slice.iter())
                .all(|(a, b)| a == b)
            {
                let user_did = did_bytes.to_base58();
                let context = self.tokenuser.lock().unwrap().get_user_context(&user_did);
                if context.is_default() || context.is_expired() {
                    self.sid_did_map.lock().unwrap().remove(sstoken);
                    println!(
                        "{} [SimpBase] The context2 of the sstoken is expired: did={}",
                        token_utils::now_string(),
                        user_did
                    );
                    String::from("Unknown")
                } else {
                    self.sid_did_map
                        .lock()
                        .unwrap()
                        .insert(sstoken.to_string(), user_did.clone());
                    user_did
                }
            } else {
                println!(
                    "{} [SimpBase] The sstoken is not validity: {}/{}",
                    token_utils::now_string(),
                    sstoken,
                    ua_hash
                );
                String::from("Unknown")
            }
        }
    }

    #[staticmethod]
    pub fn get_path_in_root_dir(did: &str, catalog: &str) -> String {
        let path_file = token_utils::get_path_in_root_dir(did, catalog);
        path_file.to_string_lossy().to_string()
    }

    pub fn get_user_path_in_root(&self, root: &str, user_did: &str) -> String {
        let root_dir = PathBuf::from(root);
        if !IdClaim::validity(user_did) {
            return root_dir.join("guest_user").to_string_lossy().to_string();
        }
        if self.absent_admin() && (self.is_guest(user_did) || user_did == self.get_local_did()) {
            return root_dir.join("Local").to_string_lossy().to_string();
        }
        let did_path = self
            .get_device_did()
            .from_base58()
            .expect("Failed to decode base58")
            .iter()
            .zip(
                user_did
                    .from_base58()
                    .expect("Failed to decode base58")
                    .iter(),
            )
            .map(|(&x, &y)| x ^ y)
            .collect::<Vec<_>>()
            .to_base58();

        if self.is_guest(user_did) {
            root_dir.join("guest_user").to_string_lossy().to_string()
        } else if self.is_admin(user_did) {
            root_dir
                .join(format!("admin_{}", did_path))
                .to_string_lossy()
                .to_string()
        } else {
            root_dir.join(did_path).to_string_lossy().to_string()
        }
    }

    pub fn set_user_base_dir(&self, user_base_dir: &str) {
        self.tokenuser
            .lock()
            .unwrap()
            .set_user_base_dir(user_base_dir)
    }

    pub fn get_path_in_user_dir(&self, did: &str, catalog: &str) -> String {
        self.tokenuser
            .lock()
            .unwrap()
            .get_path_in_user_dir(did, catalog)
    }

    pub fn get_private_paths_list(&self, did: &str, catalog: &str) -> Vec<String> {
        let catalog_paths = self.get_path_in_user_dir(did, catalog);
        let filters = &[];
        let suffixes = &[".json"];
        token_utils::filter_files(&Path::new(&catalog_paths), filters, suffixes)
    }

    pub fn get_private_paths_datas(
        &self,
        user_context: &UserContext,
        catalog: &str,
        filename: &str,
    ) -> String {
        let file_paths =
            Path::new(&self.get_path_in_user_dir(&user_context.get_did(), catalog)).join(filename);
        match file_paths.exists() {
            true => {
                let crypt_key = user_context.get_crypt_key();
                match fs::read(file_paths) {
                    Ok(raw_data) => {
                        let data = token_utils::decrypt(&raw_data, &crypt_key, 0);
                        let private_datas =
                            serde_json::from_slice(&data).unwrap_or(serde_json::json!({}));
                        private_datas.to_string()
                    }
                    Err(_) => "Unknowns".to_string(),
                }
            }
            false => "Unknowns".to_string(),
        }
    }

    pub fn get_guest_user_context(&mut self) -> UserContext {
        let guest_did = self.get_guest_did();
        self.tokenuser.lock().unwrap().get_user_context(&guest_did)
    }

    pub fn get_user_context(&mut self, did: &str) -> UserContext {
        self.tokenuser.lock().unwrap().get_user_context(did)
    }

    pub fn get_register_cert(&mut self, user_did: &str) -> String {
        self.didtoken
            .lock()
            .unwrap()
            .get_or_create_register_cert(user_did)
    }

    fn is_registered(&self, did: &str) -> bool {
        self.didtoken.lock().unwrap().is_registered(did)
    }

    fn remove_user(&self, did: &str) -> String {
        self.tokenuser.lock().unwrap().remove_user(did)
    }

    pub fn check_local_user_token(&mut self, nickname: &str, telephone: &str) -> String {
        let nickname = token_utils::truncate_nickname(nickname);
        let telephone = telephone.trim();
        if !telephone.is_empty() && !token_utils::is_valid_telephone(telephone) {
            return "unknown".to_string();
        }
        if nickname.to_lowercase().starts_with("guest") {
            return "unknown".to_string();
        }
        let telephone_for_claim = if telephone.is_empty() {
            None
        } else {
            Some(telephone.to_string())
        };
        let symbol_hash = IdClaim::get_symbol_hash_by_source(&nickname, telephone_for_claim, None);
        let (user_hash_id, _user_phrase) =
            token_utils::get_key_hash_id_and_phrase("User", &symbol_hash);
        match token_utils::exists_key_file("User", &symbol_hash) {
            true => {
                if token_utils::is_original_user_key("User", &symbol_hash) {
                    debug!("user_key exists with original phrase and should set a local phrase: {}, {}", nickname, user_hash_id);
                    "immature".to_string()
                } else {
                    "local".to_string()
                }
            }
            false => {
                let identity_file = token_utils::get_path_in_sys_key_dir(&format!(
                    "user_identity_{}.token",
                    user_hash_id
                ));
                match identity_file.exists() {
                    true => "local".to_string(),
                    false => {
                        println!("{} [SimpBase] Create local identity request: nickname={}, telephone={}, user_hash_id={}",
                                 token_utils::now_string(), nickname, telephone, user_hash_id);
                        let (user_did, _user_phrase) = self
                            .tokenuser
                            .lock()
                            .unwrap()
                            .create_user(&nickname, telephone, None, None);
                        if user_did == "Unknown" {
                            "unknown".to_string()
                        } else {
                            "immature".to_string()
                        }
                    }
                }
            }
        }
    }

    pub fn check_user_verify_code(
        &mut self,
        nickname: &str,
        telephone: &str,
        vcode: &str,
    ) -> String {
        "create".to_string()
    }

    pub fn set_phrase_and_get_context(
        &mut self,
        nickname: &str,
        telephone: &str,
        phrase: &str,
    ) -> UserContext {
        let nickname = token_utils::truncate_nickname(nickname);
        let telephone = telephone.trim();
        if !telephone.is_empty() && !token_utils::is_valid_telephone(telephone) {
            println!(
                "{} [SimpBase] The telephone number is not valid: {}, {}.",
                token_utils::now_string(),
                nickname,
                telephone
            );
            return self.get_guest_user_context();
        }
        let telephone_for_claim = if telephone.is_empty() {
            None
        } else {
            Some(telephone.to_string())
        };
        let symbol_hash = IdClaim::get_symbol_hash_by_source(&nickname, telephone_for_claim, None);
        let mut user_did = self
            .didtoken
            .lock()
            .unwrap()
            .reverse_lookup_did_by_symbol(symbol_hash);
        let symbol_hash_base64 = URL_SAFE_NO_PAD.encode(symbol_hash);
        if user_did == "Unknown" {
            let (new_user_did, _user_phrase) = self
                .tokenuser
                .lock()
                .unwrap()
                .create_user(&nickname, telephone, None, None);
            user_did = new_user_did;
        }
        if user_did == "Unknown" {
            println!(
                "{} [SimpBase] The user isn't in local: nickname={}, telephone={}, symbol={}",
                token_utils::now_string(),
                nickname,
                telephone,
                symbol_hash_base64
            );
            return self.get_guest_user_context();
        }

        let (_user_hash_id, user_phrase) =
            token_utils::get_key_hash_id_and_phrase("User", &symbol_hash);
        if token_utils::is_original_user_key("User", &symbol_hash) {
            let _ = token_utils::change_phrase_for_pem_and_identity_files(
                &symbol_hash,
                &user_phrase,
                phrase,
            );
        } else {
            println!(
                "{} [SimpBase] The user_key phrase has been changed and can not to be set: {}, {}.",
                token_utils::now_string(),
                nickname,
                user_did
            );
            return self.get_guest_user_context();
        }

        let context = self
            .tokenuser
            .lock()
            .unwrap()
            .sign_user_context(&user_did, phrase);
        if context.is_default() {
            println!(
                "{} [SimpBase] The user maybe in blacklist: {}",
                token_utils::now_string(),
                user_did
            );
            return self.get_guest_user_context();
        }
        context
    }

    pub fn get_user_context_with_phrase(
        &mut self,
        nickname: &str,
        telephone: &str,
        did: &str,
        phrase: &str,
    ) -> UserContext {
        let nickname = token_utils::truncate_nickname(nickname);
        let telephone = telephone.trim();
        if !telephone.is_empty() && !token_utils::is_valid_telephone(telephone) {
            println!(
                "{} [SimpBase] The telephone is not valid: {}",
                token_utils::now_string(),
                telephone
            );
            return self.get_guest_user_context();
        }
        let telephone_for_claim = if telephone.is_empty() {
            None
        } else {
            Some(telephone.to_string())
        };
        let symbol_hash = IdClaim::get_symbol_hash_by_source(&nickname, telephone_for_claim, None);
        let symbol_hash_base64 = URL_SAFE_NO_PAD.encode(&symbol_hash);
        let (user_hash_id, _user_phrase) =
            token_utils::get_key_hash_id_and_phrase("User", &symbol_hash);
        let mut user_did = if did.is_empty() {
            self.didtoken
                .lock()
                .unwrap()
                .reverse_lookup_did_by_symbol(symbol_hash)
        } else {
            did.to_string()
        };
        if !(token_utils::exists_and_valid_user_key(&symbol_hash, phrase) && user_did != "Unknown")
        {
            let identity_file = token_utils::get_path_in_sys_key_dir(&format!(
                "user_identity_{}.token",
                user_hash_id
            ));
            if identity_file.exists() {
                let encrypted_identity =
                    fs::read_to_string(identity_file.clone()).unwrap_or_default();
                println!(
                    "{} [SimpBase] Get user encrypted identity from local file: {}, {}, len={}",
                    token_utils::now_string(),
                    user_did,
                    symbol_hash_base64,
                    encrypted_identity.len()
                );
                user_did = self.tokenuser.lock().unwrap().import_user(
                    &symbol_hash_base64,
                    &encrypted_identity,
                    phrase,
                );
            }
        }
        if user_did != "guest" && user_did != "Unknown" {
            let context = self
                .tokenuser
                .lock()
                .unwrap()
                .sign_user_context(&user_did, phrase);
            if context.is_default() {
                println!(
                    "{} [SimpBase] The user phrase is invalid or user is blocked: {}",
                    token_utils::now_string(),
                    user_did
                );
                self.get_guest_user_context()
            } else {
                context
            }
        } else {
            self.get_guest_user_context()
        }
    }

    pub fn unbind_and_return_guest(&mut self, user_did: &str, phrase: &str) -> UserContext {
        if IdClaim::validity(user_did) {
            let claim = self.get_claim(user_did);
            if !claim.is_default() {
                // release user token and context
                if user_did != self.get_admin_did() {
                    self.tokenuser.lock().unwrap().remove_context(&user_did);
                }
            }
        }
        self.get_guest_user_context()
    }

    pub fn get_user_copy_string(&mut self, user_did: &str, phrase: &str) -> String {
        let claim = self.get_claim(user_did);
        if !claim.is_default() {
            let symbol_hash = claim.get_symbol_hash();
            let (user_hash_id, _user_phrase) =
                token_utils::get_key_hash_id_and_phrase("User", &symbol_hash);
            let identity_file = token_utils::get_path_in_sys_key_dir(&format!(
                "user_identity_{}.token",
                user_hash_id
            ));
            let encrypted_identity =
                fs::read_to_string(identity_file.clone()).unwrap_or("Unknown".to_string());
            debug!(
                "get_user_copy_string, identity_file({}), encrypted_identity: {}",
                identity_file.display(),
                encrypted_identity
            );
            let context = self.tokenuser.lock().unwrap().get_user_context(&user_did);
            let context_crypt = URL_SAFE_NO_PAD.encode(token_utils::encrypt(
                context.to_json_string().as_bytes(),
                phrase.as_bytes(),
                0,
            ));
            debug!(
                "get_user_copy_string, context_json: {}, context_crypt: {}",
                context.to_json_string(),
                context_crypt
            );
            let certificates = {
                let certificates = GlobalCerts::instance();
                let certificates = certificates
                    .lock()
                    .unwrap()
                    .filter_user_certs(&user_did, "*");
                certificates
            };
            let certificates_str = certificates
                .iter()
                .map(|(key, value)| format!("{}:{}", key, value))
                .collect::<Vec<String>>()
                .join(",");
            let certificates_str = certificates_str.replace("|", ":");
            let certificate_crypt = URL_SAFE_NO_PAD.encode(token_utils::encrypt(
                certificates_str.as_bytes(),
                phrase.as_bytes(),
                0,
            ));
            debug!(
                "get_user_copy_string, certificates_str: {}, certificate_crypt: {}",
                certificates_str, certificate_crypt
            );
            format!(
                "{}|{}|{}",
                encrypted_identity, context_crypt, certificate_crypt
            )
        } else {
            "Unknown".to_string()
        }
    }

    fn register_upstream(&mut self) -> String {
        String::new()
    }

    fn request_token_api(&mut self, api_name: &str, params: &str) -> String {
        debug!(
            "[SimpBase] local mode ignores cloud api request: api_name={}, params_len={}",
            api_name,
            params.len()
        );
        "Unknown_local_disabled".to_string()
    }

    pub fn check_ready(&self, v1: String, v2: String, v3: String, root: String) -> i32 {
        let start = Instant::now();
        let mut feedback_code = 0;
        //if !EnvData::check_basepkg(&root) {
        //    println!("[SimpBase] 程序所需基础模型包有检测异常，未完全正确安装。请检查并正确安装后，再启动程序。");
        //    feedback_code += 2;
        //}
        /*
        let mut sysinfo = self.get_sysinfo();
        loop {
            if sysinfo.pyhash != "Unknown" {
                break;
            }
            if start.elapsed() > Duration::from_secs(15) {
                println!("{} [SimpBase] 系统检测异常，继续运行会影响程序正确执行。请检查系统环境后，重新启动程序。", token_utils::now_string());
                feedback_code += 1;
                break;
            }
            thread::sleep(Duration::from_secs(1));
            sysinfo = self.get_sysinfo();
        }

        let target_pyhash= EnvData::get_pyhash(&v1, &v2, &v3);
        let check_pyhash = EnvData::get_check_pyhash(&sysinfo.pyhash.clone());
        if target_pyhash != "Unknown" && target_pyhash != check_pyhash {
            let now_sec = SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0)).as_secs();
            let pyhash_display = URL_SAFE_NO_PAD.encode(token_utils::calc_sha256(
                format!("{}-{}", sysinfo.pyhash, (now_sec/100000*100000).to_string())
                    .as_bytes()));

            println!("{} [SimpBase] 所运行程序为非官方正式版本，请正确使用开源软件，{}。", token_utils::now_string(), &pyhash_display[..16]);
            feedback_code += 4;
        }
        */

        feedback_code
    }

    pub fn get_pyhash(&self) -> String {
        let sysinfo = self.get_sysinfo();
        let pyhash = EnvData::get_check_pyhash(&sysinfo.pyhash.clone());
        pyhash
    }

    pub fn get_pyhash_key(&self, v1: String, v2: String, v3: String) -> String {
        return EnvData::get_pyhash_key(&v1, &v2, &v3);
    }

    pub fn get_claim(&self, for_did: &str) -> IdClaim {
        if for_did.is_empty() {
            debug!("get_claim in token, for_did is empty");
            return IdClaim::default();
        }
        self.didtoken.lock().unwrap().get_claim(for_did)
    }

    pub fn push_claim(&self, claim: &IdClaim) {
        self.didtoken.lock().unwrap().push_claim(claim);
    }

    pub fn pop_claim(&self, did: &str) -> IdClaim {
        self.didtoken.lock().unwrap().pop_claim(did)
    }

    pub fn import_user(
        &mut self,
        symbol_hash_base64: &str,
        encrypted_identity: &str,
        phrase: &str,
    ) -> String {
        self.tokenuser
            .lock()
            .unwrap()
            .import_user(symbol_hash_base64, encrypted_identity, phrase)
    }

    pub fn encrypt_for_did(&mut self, text: &[u8], for_did: &str, period: u64) -> String {
        self.didtoken
            .lock()
            .unwrap()
            .encrypt_for_did(text, for_did, period)
    }

    pub fn decrypt_by_did(&mut self, ctext: &str, by_did: &str, period: u64) -> String {
        self.didtoken
            .lock()
            .unwrap()
            .decrypt_by_did(ctext, by_did, period)
    }

    pub fn sign_and_issue_cert_by_admin(
        &mut self,
        item: &str,
        for_did: &str,
        for_sys_did: &str,
        memo: &str,
    ) -> (String, String) {
        self.didtoken
            .lock()
            .unwrap()
            .sign_and_issue_cert_by_admin(item, for_did, for_sys_did, memo)
    }

    pub(crate) fn sign_user_context(&mut self, did: &str, phrase: &str) -> UserContext {
        self.tokenuser
            .lock()
            .unwrap()
            .sign_user_context(did, phrase)
    }

    pub(crate) fn create_user(
        &mut self,
        nickname: &str,
        telephone: &str,
        id_card: Option<String>,
        phrase: Option<String>,
    ) -> (String, String) {
        self.tokenuser
            .lock()
            .unwrap()
            .create_user(nickname, telephone, id_card, phrase)
    }
}

pub(crate) async fn request_token_api_async(
    upstream_url: &str,
    sys_did: &str,
    dev_did: &str,
    api_name: &str,
    encoded_params: &str,
) -> String {
    debug!(
        "[Upstream] request: {}{} with params: {}, sys={}, dev={}, ver={}",
        upstream_url, api_name, encoded_params, sys_did, dev_did, TOKEN_API_VERSION
    );
    let response = match dids::REQWEST_CLIENT
        .post(format!("{}{}", upstream_url, api_name))
        .header("Sys-Did", sys_did.to_string())
        .header("Dev-Did", dev_did.to_string())
        .header("Version", TOKEN_API_VERSION.to_string())
        .body(encoded_params.to_string())
        .send()
        .await
    {
        Ok(res) => res,
        Err(e) => {
            info!(
                "Failed to send request: {} to {}{}, sys_did={}, dev_did={}",
                e, upstream_url, api_name, sys_did, dev_did
            );
            return "Unknown".to_string();
        }
    };

    // 获取状态码
    let status_code = response.status();

    // 读取响应体
    let text = match response.text().await {
        Ok(text) => text,
        Err(e) => {
            info!("Failed to read response body: {},{}", status_code, e);
            return "Unknown".to_string();
        }
    };

    // 处理响应
    debug!("[Upstream] response: {}", text);
    if status_code.is_success() {
        let result = serde_json::from_str(&text).unwrap_or("".to_string());
        debug!("[Upstream] result: {}", result);
        result
    } else {
        debug!("status_code is unsuccessful: {},{}", status_code, text);
        format!("Unknown_{}", status_code).to_string()
    }
}

async fn submit_uncompleted_request_files(upstream_url: &str, sys_did: &str, dev_did: &str) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5)); // 设置检查周期

    loop {
        interval.tick().await; // 等待下一个周期
        let user_copy_file = token_utils::get_path_in_sys_key_dir("user_copy_xxxxx.json");
        let user_copy_path = match user_copy_file.parent() {
            Some(parent) => {
                if parent.exists() {
                    parent
                } else {
                    fs::create_dir_all(parent).unwrap();
                    parent
                }
            }
            None => panic!(
                "{}",
                format!(
                    "File path does not have a parent directory: {:?}",
                    user_copy_file
                )
            ),
        };
        // 遍历目录中的所有文件
        if let Ok(mut entries) = tokio::fs::read_dir(user_copy_path).await {
            while let Some(entry) = entries.next_entry().await.transpose() {
                if let Ok(entry) = entry {
                    let file_path = entry.path();
                    if file_path.is_file() {
                        if let Some(file_name) = file_path.file_name() {
                            if let Some(file_name_str) = file_name.to_str() {
                                if let Some(method) = extract_method_from_filename(file_name_str) {
                                    if let Ok(content) = tokio::fs::read_to_string(&file_path).await
                                    {
                                        debug!(
                                            "submit uncompleted request file: method={}, {}",
                                            method,
                                            file_path.display()
                                        );
                                        let result = request_token_api_async(
                                            upstream_url,
                                            sys_did,
                                            dev_did,
                                            &method,
                                            &content,
                                        )
                                        .await;
                                        if result != "Unknown" {
                                            tokio::fs::remove_file(&file_path)
                                                .await
                                                .expect("remove user copy file failed");
                                            debug!(
                                                "remove the uncompleted request file: {}",
                                                file_path.display()
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn extract_method_from_filename(file_name: &str) -> Option<String> {
    let re = regex::Regex::new(r"^(.+?)_([a-zA-Z0-9]{29})_uncompleted\.json$").unwrap();
    if let Some(captures) = re.captures(file_name) {
        if let Some(method) = captures.get(1) {
            return Some(method.as_str().to_string());
        }
    }
    None
}

async fn sync_upstream(
    sys_did: &str,
    dev_did: &str,
    upstream_did: String,
    entry_point: Arc<tokio::sync::Mutex<DidEntryPoint>>,
    online_users: Arc<tokio::sync::Mutex<OnlineUsers>>,
    message_queue: Arc<MessageQueue>,
) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

    loop {
        let mut upstream_did = upstream_did.clone();
        let result_string = {
            let mut request = json!({});
            let online_users_list = {
                let users_guard = online_users.lock().await;
                users_guard.get_full_list()
            };
            let last_timestamp = message_queue
                .get_last_timestamp(sys_did)
                .unwrap_or_else(|| 0u64);
            request["online_users"] = serde_json::to_value(online_users_list).unwrap_or(json!(""));
            request["msg_timestamp"] = serde_json::to_value(last_timestamp).unwrap_or(json!(0u64));
            let params = serde_json::to_string(&request).unwrap_or("{}".to_string());
            let upstream_url = {
                let ep = entry_point.lock().await;
                ep.get_entry_point(&upstream_did.clone())
            };
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(5),
                request_token_api_async(&upstream_url, sys_did, dev_did, "ping", &params),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => "Unknown".to_string(),
            }
        };

        debug!(
            "{} [Upstream] {} ping upstream node: {}",
            token_utils::now_string(),
            sys_did,
            result_string
        );

        if result_string != "Unknown" {
            let mut ping_vars = serde_json::from_str::<HashMap<String, String>>(&result_string)
                .unwrap_or_else(|_| HashMap::new());
            if let Some(user_online) = ping_vars.get("user_online") {
                let user_online_array: Vec<&str> = user_online.split(":").collect();
                if user_online_array.len() >= 3 {
                    let nodes = user_online_array[0].parse().unwrap_or(1);
                    let users = user_online_array[1].parse().unwrap_or(1);
                    let top_list = user_online_array[2].to_string();
                    if nodes > 1 && users > 1 {
                        let mut users_guard = online_users.lock().await;
                        users_guard.set_nodes_users(nodes, users, top_list.clone());
                        debug!(
                            "{} [Upstream] set_nodes_users: {}:{}:{}",
                            token_utils::now_string(),
                            nodes,
                            users,
                            top_list
                        );
                    } else if nodes == 0 && users == 0 {
                        debug!(
                            "{} [Upstream] get null nodes_users: {}:{}:{}",
                            token_utils::now_string(),
                            nodes,
                            users,
                            top_list
                        );
                        let claims = GlobalClaims::instance();
                        let (local_claim, device_claim) = {
                            let mut claims = claims.lock().unwrap();
                            (
                                claims.get_claim_from_local(sys_did),
                                claims.get_claim_from_local(dev_did),
                            )
                        };
                        let last_timestamp = message_queue
                            .get_last_timestamp(sys_did)
                            .unwrap_or_else(|| 0u64);
                        let mut request = json!({});
                        request["system_claim"] =
                            serde_json::to_value(local_claim).unwrap_or(json!(""));
                        request["device_claim"] =
                            serde_json::to_value(device_claim).unwrap_or(json!(""));
                        request["msg_timestamp"] =
                            serde_json::to_value(last_timestamp).unwrap_or(json!(0u64));

                        let params = serde_json::to_string(&request).unwrap_or("{}".to_string());
                        let upstream_url = {
                            let ep = entry_point.lock().await;
                            ep.get_entry_point(dids::TOKEN_ENTRYPOINT_DID)
                        };
                        let response = request_token_api_async(
                            &upstream_url,
                            &sys_did,
                            &dev_did,
                            "register2",
                            &params,
                        )
                        .await;
                        ping_vars = serde_json::from_str::<HashMap<String, String>>(&response)
                            .unwrap_or_else(|_| HashMap::new());
                        debug!(
                            "{} [Upstream] repair ping: {}",
                            token_utils::now_string(),
                            response
                        );
                        upstream_did = if let Some(new_did) = ping_vars.get("upstream_did") {
                            new_did.clone()
                        } else {
                            upstream_did.clone()
                        };
                    }
                }
            }

            if let Some(message_list) = ping_vars.get("message_list") {
                message_queue.push_messages(&sys_did, message_list.to_string());
            }
        }

        interval.tick().await;
    }
}
