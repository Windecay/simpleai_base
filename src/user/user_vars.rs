use chrono::format;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, trace, warn};
use warp::filters::body::form;

use crate::dids::{self, tokendb::TokenDB, DidToken};
use crate::user::TokenUser;

lazy_static::lazy_static! {
    static ref ADMIN_DEFAULT: Arc<RwLock<AdminDefault>> = Arc::new(RwLock::new(AdminDefault::new()));
    static ref GLOBEL_LOCAL_VARS: Arc<RwLock<GlobalLocalVars>> = Arc::new(RwLock::new(GlobalLocalVars::new()));
}

#[derive(Clone, Debug)]
pub struct GlobalLocalVars {
    sys_did: String,
    device_did: String,
    guest_did: String,
    admin_did: String,
    token_db: Arc<RwLock<TokenDB>>, //HashMap<global|admin|{did}_{key}, String>,
    didtoken: Arc<Mutex<DidToken>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserAccessRecord {
    pub did: String,
    pub nickname: String,
    pub status: String,
    pub can_generate: bool,
    #[serde(default)]
    pub can_download_models: bool,
    pub updated_at: u64,
}

impl GlobalLocalVars {
    pub fn instance() -> Arc<RwLock<GlobalLocalVars>> {
        GLOBEL_LOCAL_VARS.clone()
    }

    pub fn new() -> Self {
        let didtoken = DidToken::instance();
        let (sys_did, device_did, guest_did, admin_did, token_db) = {
            let didtoken = didtoken.lock().unwrap();
            (
                didtoken.get_sys_did(),
                didtoken.get_device_did(),
                didtoken.get_guest_did(),
                didtoken.get_admin_did(),
                didtoken.get_token_db(),
            )
        };

        Self {
            sys_did,
            device_did,
            guest_did,
            admin_did,
            token_db,
            didtoken,
        }
    }

    pub fn get_admin_did(&self) -> String {
        self.admin_did.clone()
    }
    pub(crate) fn set_admin_did(&mut self, admin_did: &str) {
        self.admin_did = admin_did.to_string();
    }

    pub fn get_global_vars(&self, key: &str, default: &str) -> String {
        let key = format!("global_{}", key);
        let token_db = match self.token_db.read() {
            Ok(guard) => guard,
            Err(e) => {
                error!("获取全局变量读锁失败: key={}, error={:?}", key, e);
                return default.to_string();
            }
        };
        let vars_value = token_db.get("global_local_vars", &key);
        if !vars_value.is_empty() && vars_value != "Unknown" {
            vars_value
        } else {
            default.to_string()
        }
    }

    pub fn put_global_var(&mut self, key: &str, value: &str) {
        let key = format!("global_{}", key);
        let mut token_db = match self.token_db.write() {
            Ok(guard) => guard,
            Err(e) => {
                error!("获取全局变量写锁失败: key={}, error={:?}", key, e);
                return;
            }
        };
        token_db.insert("global_local_vars", &key, value);
    }

    pub fn get_global_vars_json(&self) -> String {
        let prefix = "global_";
        let global_vars: HashMap<String, String> = HashMap::new();
        let token_db = match self.token_db.write() {
            Ok(guard) => guard,
            Err(e) => {
                error!("获取全局变量写锁失败: error={:?}", e);
                return serde_json::to_string(&global_vars).unwrap_or_default();
            }
        };
        let global_vars = token_db.scan_prefix("global_local_vars", prefix);
        match serde_json::to_string(&global_vars) {
            Ok(json) => json,
            Err(e) => {
                error!("全局变量JSON序列化失败: error={:?}", e);
                String::new()
            }
        }
    }

    pub(crate) fn get_local_admin_vars(&self, key: &str) -> String {
        let admin_key = format!("admin_{}_{}", self.sys_did, key);
        let admin_did = self.get_admin_did();
        self.get_local_vars(&admin_key, "default", &admin_did)
    }

    pub(crate) fn get_local_vars(&self, key: &str, default: &str, user_did: &str) -> String {
        // 1. 确定变量类型和键名
        let is_admin_var = key.starts_with("admin_");
        let (local_did, local_key) = if is_admin_var {
            (self.get_admin_did(), key.to_string())
        } else {
            (
                user_did.to_string(),
                format!("{}_{}_{}", user_did, self.sys_did, key),
            )
        };

        // 2. 从存储中获取原始值
        let raw_value = match self.token_db.read() {
            Ok(guard) => guard.get("global_local_vars", &local_key),
            Err(e) => {
                error!("从存储中获取原始值: key={}, error={:?}", local_key, e);
                "Unknown".to_string()
            }
        };
        // 3. 处理特殊值情况
        if raw_value == "Default" || raw_value == "Unknown" {
            return if is_admin_var {
                let admin_key_prefix = format!("admin_{}_", self.sys_did);
                let default_key_name = key.trim_start_matches(admin_key_prefix.as_str());
                AdminDefault::instance()
                    .read()
                    .unwrap()
                    .get(default_key_name)
            } else {
                default.to_string()
            };
        }

        // 4. 处理管理员变量解密
        if is_admin_var {
            if local_did.is_empty() {
                return "Unknown".to_string();
            }
            let admin_value = self
                .didtoken
                .lock()
                .unwrap()
                .decrypt_by_did(&raw_value, &local_did, 0);
            debug!("get and decode admin_value: {}", admin_value);
            if admin_value.is_empty() || admin_value == "Unknown" {
                match self.token_db.write() {
                    Ok(guard) => guard.remove("global_local_vars", &local_key),
                    Err(e) => {
                        error!("获取全局变量写锁失败: error={:?}", e);
                        false
                    }
                };
                let admin_default = {
                    let admin_key_prefix = format!("admin_{}_", self.sys_did);
                    let default_key_name = key.trim_start_matches(admin_key_prefix.as_str());
                    AdminDefault::instance()
                        .read()
                        .unwrap()
                        .get(default_key_name)
                };
                return admin_default;
            }
            return admin_value;
        }

        raw_value
    }

    pub(crate) fn set_local_admin_vars(&mut self, key: &str, value: &str) {
        let admin_key = format!("admin_{}_{}", self.sys_did, key);
        let admin_did = self.get_admin_did();
        self.set_local_vars(&admin_key, value, &admin_did);
    }

    pub(crate) fn set_local_vars(&mut self, key: &str, value: &str, user_did: &str) {
        let is_admin_var = key.starts_with("admin_");
        let admin_did = self.get_admin_did();
        if is_admin_var && admin_did != user_did {
            println!("非管理员用户 {} 在尝试设置管理员变量 {}", user_did, key);
            return;
        }
        let (local_key, local_value) = if is_admin_var {
            // 管理员变量需要加密
            let encrypted_value =
                self.didtoken
                    .lock()
                    .unwrap()
                    .encrypt_for_did(&value.as_bytes(), &admin_did, 0);
            let admin_key = key.trim_start_matches("admin_");
            let admin_key = format!("admin_{}_{}", self.sys_did, admin_key);
            (admin_key.to_string(), encrypted_value)
        } else {
            // 普通用户变量
            (
                format!("{}_{}_{}", user_did, self.sys_did, key),
                value.to_string(),
            )
        };
        let _ = match self.token_db.write() {
            Ok(mut guard) => guard.insert("global_local_vars", &local_key, &local_value),
            Err(e) => {
                error!("获取global_local_vars写锁失败: {:?}", e);
                false
            }
        };
    }

    pub fn set_local_vars_for_guest(&mut self, key: &str, value: &str, user_did: &str) {
        let admin = self.didtoken.lock().unwrap().get_admin_did();
        let guest = self.guest_did.clone();
        if admin != user_did {
            return;
        }
        let local_key = format!("{}_{}_{}", guest, self.sys_did, key);
        let local_value = value.to_string();
        let _ = match self.token_db.write() {
            Ok(mut guard) => guard.insert("global_local_vars", &local_key, &local_value),
            Err(e) => {
                error!("获取global_local_vars写锁失败: {:?}", e);
                false
            }
        };
    }

    pub fn get_message_list(&self, user_did: &str) -> String {
        let key = format!("msg_list_{}_{}", self.sys_did, user_did);
        let result = match self.token_db.read() {
            Ok(guard) => guard.get("global_local_vars", &key),
            Err(e) => {
                error!("获取消息列表读锁失败: user_did={}, error={:?}", user_did, e);
                String::new()
            }
        };
        result
    }

    pub fn set_message_list(&mut self, user_did: &str, message_list: &str) {
        let key = format!("msg_list_{}_{}", self.sys_did, user_did);
        let _ = match self.token_db.write() {
            Ok(guard) => guard.insert("global_local_vars", &key, &message_list),
            Err(e) => {
                error!("获取消息列表写锁失败: user_did={}, error={:?}", user_did, e);
                false
            }
        };
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs()
    }

    fn list_contains(list: &str, did: &str) -> bool {
        if did.is_empty() {
            return false;
        }
        list.split(',').map(|s| s.trim()).any(|item| item == did)
    }

    fn list_add(list: &str, did: &str) -> String {
        let mut values: Vec<String> = list
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if !values.iter().any(|item| item == did) && !did.is_empty() {
            values.push(did.to_string());
        }
        values.join(",")
    }

    fn list_remove(list: &str, did: &str) -> String {
        list.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && *s != did)
            .collect::<Vec<&str>>()
            .join(",")
    }

    fn user_access_key(&self, did: &str) -> String {
        format!("user_access_{}_{}", self.sys_did, did)
    }

    pub(crate) fn set_user_access_record(
        &mut self,
        did: &str,
        nickname: &str,
        status: &str,
        can_generate: bool,
        can_download_models: bool,
    ) {
        if did.is_empty() {
            return;
        }
        let record = UserAccessRecord {
            did: did.to_string(),
            nickname: nickname.to_string(),
            status: status.to_string(),
            can_generate,
            can_download_models,
            updated_at: Self::now_secs(),
        };
        let key = self.user_access_key(did);
        let value = serde_json::to_string(&record).unwrap_or_else(|_| "{}".to_string());
        let _ = match self.token_db.write() {
            Ok(guard) => guard.insert("global_local_vars", &key, &value),
            Err(e) => {
                error!("写入用户权限失败: did={}, error={:?}", did, e);
                false
            }
        };
    }

    pub(crate) fn get_user_access_record(&self, did: &str) -> Option<UserAccessRecord> {
        if did.is_empty() {
            return None;
        }
        let key = self.user_access_key(did);
        let raw_value = match self.token_db.read() {
            Ok(guard) => guard.get("global_local_vars", &key),
            Err(e) => {
                error!("读取用户权限失败: did={}, error={:?}", did, e);
                "Unknown".to_string()
            }
        };
        if raw_value.is_empty() || raw_value == "Unknown" {
            return None;
        }
        serde_json::from_str::<UserAccessRecord>(&raw_value).ok()
    }

    pub(crate) fn get_user_access_list(&self) -> String {
        let prefix = format!("user_access_{}_", self.sys_did);
        let records = match self.token_db.read() {
            Ok(guard) => guard.scan_prefix("global_local_vars", &prefix),
            Err(e) => {
                error!("读取用户权限列表失败: error={:?}", e);
                HashMap::new()
            }
        };
        let mut result: Vec<UserAccessRecord> = records
            .values()
            .filter_map(|value| serde_json::from_str::<UserAccessRecord>(value).ok())
            .collect();
        result.sort_by(|a, b| a.updated_at.cmp(&b.updated_at).then(a.did.cmp(&b.did)));
        serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string())
    }

    fn get_default_user_can_generate(&self) -> bool {
        self.get_local_admin_vars("default_user_can_generate") == "True"
    }

    fn get_default_user_can_download_models(&self) -> bool {
        self.get_local_admin_vars("default_user_can_download_models") == "True"
    }

    pub(crate) fn approve_user(&mut self, did: &str, nickname: &str, can_generate: bool) {
        let can_download_models = self.get_default_user_can_download_models();
        self.approve_user_with_permissions(did, nickname, can_generate, can_download_models);
    }

    pub(crate) fn approve_user_with_permissions(
        &mut self,
        did: &str,
        nickname: &str,
        can_generate: bool,
        can_download_models: bool,
    ) {
        self.remove_pending_did(did, "web");
        self.add_allowed_did(did, "web");
        self.set_user_access_record(did, nickname, "allowed", can_generate, can_download_models);
    }

    pub(crate) fn reject_user(&mut self, did: &str, nickname: &str) {
        self.remove_pending_did(did, "web");
        self.remove_allowed_did(did, "web");
        self.set_user_access_record(did, nickname, "blocked", false, false);
    }

    pub(crate) fn set_user_can_generate(&mut self, did: &str, can_generate: bool) {
        let record = self.get_user_access_record(did);
        let nickname = record
            .as_ref()
            .map(|record| record.nickname.clone())
            .unwrap_or_default();
        let status = record
            .as_ref()
            .map(|record| record.status.as_str())
            .unwrap_or_else(|| {
                if self.is_allowed_did(did, "web") {
                    "allowed"
                } else {
                    "pending"
                }
            });
        let can_download_models = record
            .as_ref()
            .map(|record| record.can_download_models)
            .unwrap_or_else(|| self.get_default_user_can_download_models());
        self.set_user_access_record(did, &nickname, status, can_generate, can_download_models);
    }

    pub(crate) fn set_user_can_download_models(&mut self, did: &str, can_download_models: bool) {
        let record = self.get_user_access_record(did);
        let nickname = record
            .as_ref()
            .map(|record| record.nickname.clone())
            .unwrap_or_default();
        let status = record
            .as_ref()
            .map(|record| record.status.as_str())
            .unwrap_or_else(|| {
                if self.is_allowed_did(did, "web") {
                    "allowed"
                } else {
                    "pending"
                }
            });
        let can_generate = record
            .as_ref()
            .map(|record| record.can_generate)
            .unwrap_or_else(|| self.get_default_user_can_generate());
        self.set_user_access_record(did, &nickname, status, can_generate, can_download_models);
    }

    pub(crate) fn set_guest_can_generate(&mut self, can_generate: bool) {
        self.set_local_admin_vars(
            "guest_can_generate",
            if can_generate { "True" } else { "False" },
        );
    }

    pub(crate) fn get_guest_can_generate(&self) -> bool {
        self.get_local_admin_vars("guest_can_generate") == "True"
    }

    pub(crate) fn set_guest_can_download_models(&mut self, can_download_models: bool) {
        self.set_local_admin_vars(
            "guest_can_download_models",
            if can_download_models { "True" } else { "False" },
        );
    }

    pub(crate) fn get_guest_can_download_models(&self) -> bool {
        self.get_local_admin_vars("guest_can_download_models") == "True"
    }

    pub(crate) fn can_user_generate(&self, did: &str) -> bool {
        let admin_did = self.get_admin_did();
        if admin_did.is_empty() {
            return true;
        }
        if did == admin_did {
            return true;
        }
        if did == self.guest_did {
            return self.get_guest_can_generate();
        }
        if let Some(record) = self.get_user_access_record(did) {
            return record.status == "allowed" && record.can_generate;
        }
        self.is_allowed_did(did, "web")
    }

    pub(crate) fn can_user_download_models(&self, did: &str) -> bool {
        let admin_did = self.get_admin_did();
        if admin_did.is_empty() {
            return true;
        }
        if did == admin_did {
            return true;
        }
        if did == self.guest_did {
            return self.get_guest_can_download_models();
        }
        if let Some(record) = self.get_user_access_record(did) {
            return record.status == "allowed" && record.can_download_models;
        }
        self.is_allowed_did(did, "web") && self.get_default_user_can_download_models()
    }

    pub(crate) fn is_allowed_did(&self, did: &str, way: &str) -> bool {
        if way == "web" || way == "p2p" {
            Self::list_contains(
                &self.get_local_admin_vars(&format!("{way}_in_did_list")),
                did,
            )
        } else {
            false
        }
    }

    pub(crate) fn get_allowed_did_list(&self, way: &str) -> String {
        if way != "web" && way != "p2p" {
            return String::new();
        }
        let list_key = format!("{way}_in_did_list");
        self.get_local_admin_vars(&list_key)
    }

    pub(crate) fn add_allowed_did(&mut self, did: &str, way: &str) {
        if way != "web" && way != "p2p" {
            return;
        }
        let list_key = format!("{way}_in_did_list");

        let did_list = Self::list_add(&self.get_local_admin_vars(&list_key), did);
        self.set_local_admin_vars(&list_key, &did_list);
    }

    pub(crate) fn remove_allowed_did(&mut self, did: &str, way: &str) {
        if way != "web" && way != "p2p" {
            return;
        }
        let list_key = format!("{way}_in_did_list");

        let did_list = self.get_local_admin_vars(&list_key);
        if did_list.is_empty() {
            return;
        } else if Self::list_contains(&did_list, did) {
            let new_did_list = Self::list_remove(&did_list, did);
            self.set_local_admin_vars(&list_key, &new_did_list);
        }
    }

    pub(crate) fn is_pending_did(&self, did: &str, way: &str) -> bool {
        if way == "web" || way == "p2p" {
            Self::list_contains(
                &self.get_local_admin_vars(&format!("{way}_pending_did_list")),
                did,
            )
        } else {
            false
        }
    }

    pub(crate) fn get_pending_did_list(&self, way: &str) -> String {
        if way != "web" && way != "p2p" {
            return String::new();
        }
        let list_key = format!("{way}_pending_did_list");
        self.get_local_admin_vars(&list_key)
    }

    pub(crate) fn add_pending_did(&mut self, did: &str, way: &str) {
        if way != "web" && way != "p2p" {
            return;
        }
        let list_key = format!("{way}_pending_did_list");

        let did_list = Self::list_add(&self.get_local_admin_vars(&list_key), did);
        self.set_local_admin_vars(&list_key, &did_list);
    }

    pub(crate) fn remove_pending_did(&mut self, did: &str, way: &str) {
        if way != "web" && way != "p2p" {
            return;
        }
        let list_key = format!("{way}_pending_did_list");

        let did_list = self.get_local_admin_vars(&list_key);
        if did_list.is_empty() {
            return;
        } else if Self::list_contains(&did_list, did) {
            let new_did_list = Self::list_remove(&did_list, did);
            self.set_local_admin_vars(&list_key, &new_did_list);
        }
    }

    pub(crate) fn pending_to_allowed_did(&mut self, did: &str, way: &str) {
        self.remove_pending_did(did, way);
        self.add_allowed_did(did, way);
    }
}

pub struct AdminDefault {
    data: HashMap<String, String>,
}
impl AdminDefault {
    pub fn instance() -> Arc<RwLock<AdminDefault>> {
        ADMIN_DEFAULT.clone()
    }
    pub fn new() -> Self {
        let mut data = HashMap::new();
        data.insert("comfyd_active_checkbox".to_string(), "True".to_string());
        data.insert("fast_comfyd_checkbox".to_string(), "False".to_string());
        data.insert("reserved_vram".to_string(), "0".to_string());
        data.insert("vlm_checkbox".to_string(), "False".to_string());
        data.insert(
            "vlm_version".to_string(),
            "Qwen3.5-9B-abliterated-Q4_K_M".to_string(),
        );
        data.insert("advanced_logs".to_string(), "False".to_string());
        data.insert("wavespeed_strength".to_string(), "0.12".to_string());
        data.insert("translation_methods".to_string(), "Third APIs".to_string());
        data.insert("topbar_button_quantity".to_string(), "10".to_string());
        data.insert("p2p_active_checkbox".to_string(), "False".to_string());
        data.insert("p2p_remote_process".to_string(), "Disable".to_string());
        data.insert("p2p_in_did_list".to_string(), "".to_string());
        data.insert("p2p_out_did_list".to_string(), "".to_string());
        data.insert("guest_can_generate".to_string(), "False".to_string());
        data.insert("guest_can_download_models".to_string(), "False".to_string());
        data.insert("default_user_can_generate".to_string(), "True".to_string());
        data.insert(
            "default_user_can_download_models".to_string(),
            "False".to_string(),
        );
        Self { data }
    }
    pub fn get(&self, key: &str) -> String {
        self.data
            .get(key)
            .unwrap_or(&"None".to_string())
            .to_string()
    }
    pub fn insert(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
    pub fn remove(&mut self, key: &str) -> String {
        self.data.remove(key).unwrap_or("None".to_string())
    }
}
