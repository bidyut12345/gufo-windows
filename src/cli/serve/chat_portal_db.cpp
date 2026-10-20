#include "src/cli/serve/chat_portal_db.hpp"

#include <sqlite3.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "src/core/crypto/sha256.hpp"

namespace gufo::server {

std::string SanitizeFolderName(const std::string& raw) {
  std::string clean;
  clean.reserve(raw.size());
  for (char c : raw) {
    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
        c == '"' || c == '<' || c == '>' || c == '|' || c == '\n' ||
        c == '\r' || c == '\t') {
      clean += '_';
    } else {
      clean += c;
    }
  }
  while (!clean.empty() && (clean.front() == ' ' || clean.front() == '.')) {
    clean.erase(clean.begin());
  }
  while (!clean.empty() && (clean.back() == ' ' || clean.back() == '.')) {
    clean.pop_back();
  }
  if (clean.empty()) {
    clean = "Conversation";
  }
  if (clean.size() > 60) {
    clean.resize(60);
  }
  return clean;
}

std::filesystem::path GetExecutableDirectory() {
#if defined(_WIN32)
  wchar_t buffer[MAX_PATH];
  DWORD len = GetModuleFileNameW(NULL, buffer, MAX_PATH);
  if (len > 0 && len < MAX_PATH) {
    return std::filesystem::path(buffer).parent_path();
  }
#elif defined(__linux__)
  char buffer[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len != -1) {
    buffer[len] = '\0';
    return std::filesystem::path(buffer).parent_path();
  }
#endif
  std::error_code ec;
  return std::filesystem::current_path(ec);
}

std::filesystem::path GetConversationDefaultWorkingDir(const std::string& conv_id, const std::string& title) {
  std::string folder_name = conv_id + " - " + SanitizeFolderName(title);
  auto p = GetExecutableDirectory() / "UserData" / folder_name;
  std::error_code ec;
  std::filesystem::create_directories(p, ec);
  return p;
}

namespace {

std::string CurrentIsoTime() {
  const auto now = std::chrono::system_clock::now();
  const auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  std::tm buf{};
#if defined(_WIN32)
  gmtime_s(&buf, &in_time_t);
#else
  gmtime_r(&in_time_t, &buf);
#endif
  ss << std::put_time(&buf, "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

std::string RandomHex(std::size_t num_bytes) {
  static thread_local std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<unsigned int> dist(0, 255);
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < num_bytes; ++i) {
    ss << std::setw(2) << dist(rng);
  }
  return ss.str();
}

std::string HashPassword(const std::string& salt, const std::string& password) {
  const std::string combined = salt + ":" + password;
  return crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(combined.data()), combined.size()));
}

struct Stmt {
  sqlite3_stmt* stmt{nullptr};

  Stmt(sqlite3* db, const char* sql) {
    if (db) {
      sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    }
  }

  ~Stmt() {
    if (stmt) {
      sqlite3_finalize(stmt);
    }
  }

  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;
  Stmt(Stmt&& o) noexcept : stmt(o.stmt) { o.stmt = nullptr; }
  Stmt& operator=(Stmt&& o) noexcept {
    if (this != &o) {
      if (stmt) sqlite3_finalize(stmt);
      stmt = o.stmt;
      o.stmt = nullptr;
    }
    return *this;
  }

  operator sqlite3_stmt*() const { return stmt; }
  bool valid() const { return stmt != nullptr; }

  void bind_text(int idx, const std::string& s) {
    sqlite3_bind_text(stmt, idx, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
  }

  void bind_double(int idx, double val) {
    sqlite3_bind_double(stmt, idx, val);
  }

  void bind_int(int idx, int val) {
    sqlite3_bind_int(stmt, idx, val);
  }

  std::string column_text(int col) const {
    const auto* p = sqlite3_column_text(stmt, col);
    return p ? reinterpret_cast<const char*>(p) : std::string{};
  }

  double column_double(int col) const {
    return sqlite3_column_double(stmt, col);
  }

  int column_int(int col) const {
    return sqlite3_column_int(stmt, col);
  }
};

}  // namespace

ChatDatabase& ChatDatabase::Instance() {
  static ChatDatabase instance;
  return instance;
}

std::filesystem::path ChatDatabase::DefaultDbPath() {
  if (const char* env_path = std::getenv("GUFO_CHAT_DB")) {
    if (*env_path) {
      return std::filesystem::path(env_path);
    }
  }

#if defined(_WIN32)
  if (const char* appdata = std::getenv("LOCALAPPDATA")) {
    const auto dir = std::filesystem::path(appdata) / "gufo";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir / "chat_portal.db";
  }
#else
  if (const char* home = std::getenv("HOME")) {
    const auto dir = std::filesystem::path(home) / ".local" / "share" / "gufo";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir / "chat_portal.db";
  }
#endif

  return std::filesystem::current_path() / "gufo_chat_portal.db";
}

ChatDatabase::ChatDatabase(std::filesystem::path db_path)
    : path_(db_path.empty() ? DefaultDbPath() : std::move(db_path)) {
  std::error_code ec;
  if (path_.has_parent_path()) {
    std::filesystem::create_directories(path_.parent_path(), ec);
  }

  std::lock_guard lock(mutex_);
  const int rc = sqlite3_open_v2(
      path_.string().c_str(), &db_,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      nullptr);
  if (rc != SQLITE_OK) {
    if (db_) {
      sqlite3_close(db_);
      db_ = nullptr;
    }
    return;
  }
  InitSchema();
}

ChatDatabase::~ChatDatabase() {
  std::lock_guard lock(mutex_);
  if (db_) {
    sqlite3_close_v2(db_);
    db_ = nullptr;
  }
}

void ChatDatabase::InitSchema() {
  if (!db_) return;

  const char* schema = R"sql(
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      username TEXT UNIQUE NOT NULL COLLATE NOCASE,
      password_hash TEXT NOT NULL,
      salt TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
      token TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      expires_at TEXT,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      title TEXT NOT NULL,
      model TEXT NOT NULL,
      system_prompt TEXT,
      working_directory TEXT DEFAULT '',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      reasoning_content TEXT,
      created_at TEXT NOT NULL,
      order_idx INTEGER NOT NULL,
      FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, order_idx);
    CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);

    CREATE TABLE IF NOT EXISTS user_settings (
      user_id TEXT PRIMARY KEY,
      temperature REAL DEFAULT 0.6,
      top_p REAL DEFAULT 0.95,
      max_tokens INTEGER DEFAULT 4096,
      system_prompt TEXT,
      working_directory TEXT DEFAULT '',
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS mcp_servers (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      url TEXT NOT NULL,
      transport TEXT DEFAULT 'sse',
      headers_json TEXT DEFAULT '{}',
      tools_cache_json TEXT DEFAULT '[]',
      enabled INTEGER DEFAULT 1,
      created_at TEXT NOT NULL,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS custom_tools (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      description TEXT NOT NULL,
      schema_json TEXT NOT NULL,
      handler_url TEXT DEFAULT '',
      enabled INTEGER DEFAULT 1,
      created_at TEXT NOT NULL,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS user_tools_config (
      user_id TEXT PRIMARY KEY,
      inbuilt_tools_json TEXT NOT NULL,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    INSERT OR IGNORE INTO users (id, username, password_hash, salt, created_at)
    VALUES ('u_guest', 'Guest', '', '', '2026-01-01T00:00:00Z');
  )sql";

  char* err_msg = nullptr;
  sqlite3_exec(db_, schema, nullptr, nullptr, &err_msg);
  if (err_msg) {
    sqlite3_free(err_msg);
  }

  // Gracefully migrate existing databases if working_directory column is missing
  sqlite3_exec(
      db_,
      "ALTER TABLE user_settings ADD COLUMN working_directory TEXT DEFAULT '';",
      nullptr, nullptr, nullptr);
  sqlite3_exec(
      db_,
      "ALTER TABLE conversations ADD COLUMN working_directory TEXT DEFAULT '';",
      nullptr, nullptr, nullptr);
}

ChatDatabase::AuthResult ChatDatabase::Register(const std::string& username,
                                               const std::string& password) {
  std::lock_guard lock(mutex_);
  if (!db_) {
    return {false, "Database connection not open.", "", {}};
  }

  if (username.empty() || username.size() > 64) {
    return {false, "Username must be between 1 and 64 characters.", "", {}};
  }
  if (password.empty()) {
    return {false, "Password cannot be empty.", "", {}};
  }

  // Check existing user
  {
    Stmt check(db_, "SELECT id FROM users WHERE username = ? COLLATE NOCASE;");
    if (!check.valid()) {
      return {false, "Internal database error.", "", {}};
    }
    check.bind_text(1, username);
    if (sqlite3_step(check) == SQLITE_ROW) {
      return {false, "Username already taken.", "", {}};
    }
  }

  UserRecord user;
  user.id = "usr_" + RandomHex(8);
  user.username = username;
  user.salt = RandomHex(16);
  user.password_hash = HashPassword(user.salt, password);
  user.created_at = CurrentIsoTime();

  {
    Stmt ins(db_,
             "INSERT INTO users (id, username, password_hash, salt, created_at) "
             "VALUES (?, ?, ?, ?, ?);");
    if (!ins.valid()) {
      return {false, "Internal database error.", "", {}};
    }
    ins.bind_text(1, user.id);
    ins.bind_text(2, user.username);
    ins.bind_text(3, user.password_hash);
    ins.bind_text(4, user.salt);
    ins.bind_text(5, user.created_at);
    if (sqlite3_step(ins) != SQLITE_DONE) {
      return {false, "Failed to create user record.", "", {}};
    }
  }

  // Generate session token
  const std::string token = "gufotok_" + RandomHex(24);
  const std::string token_created = CurrentIsoTime();
  {
    Stmt ins_sess(db_,
                  "INSERT INTO sessions (token, user_id, created_at, expires_at) "
                  "VALUES (?, ?, ?, '');");
    if (ins_sess.valid()) {
      ins_sess.bind_text(1, token);
      ins_sess.bind_text(2, user.id);
      ins_sess.bind_text(3, token_created);
      sqlite3_step(ins_sess);
    }
  }

  return {true, "", token, std::move(user)};
}

ChatDatabase::AuthResult ChatDatabase::Login(const std::string& username,
                                            const std::string& password) {
  std::lock_guard lock(mutex_);
  if (!db_) {
    return {false, "Database connection not open.", "", {}};
  }

  UserRecord user;
  {
    Stmt sel(db_,
             "SELECT id, username, password_hash, salt, created_at FROM users "
             "WHERE username = ? COLLATE NOCASE;");
    if (!sel.valid()) {
      return {false, "Internal database error.", "", {}};
    }
    sel.bind_text(1, username);
    if (sqlite3_step(sel) != SQLITE_ROW) {
      return {false, "Invalid username or password.", "", {}};
    }

    user.id = sel.column_text(0);
    user.username = sel.column_text(1);
    user.password_hash = sel.column_text(2);
    user.salt = sel.column_text(3);
    user.created_at = sel.column_text(4);
  }

  const std::string hashed = HashPassword(user.salt, password);
  if (hashed != user.password_hash) {
    return {false, "Invalid username or password.", "", {}};
  }

  const std::string token = "gufotok_" + RandomHex(24);
  const std::string token_created = CurrentIsoTime();
  {
    Stmt ins_sess(db_,
                  "INSERT INTO sessions (token, user_id, created_at, expires_at) "
                  "VALUES (?, ?, ?, '');");
    if (ins_sess.valid()) {
      ins_sess.bind_text(1, token);
      ins_sess.bind_text(2, user.id);
      ins_sess.bind_text(3, token_created);
      sqlite3_step(ins_sess);
    }
  }

  return {true, "", token, std::move(user)};
}

std::optional<UserRecord> ChatDatabase::AuthenticateToken(const std::string& token) {
  if (token.empty()) return std::nullopt;

  std::lock_guard lock(mutex_);
  if (!db_) return std::nullopt;

  Stmt sel(db_,
           "SELECT u.id, u.username, u.password_hash, u.salt, u.created_at "
           "FROM sessions s JOIN users u ON s.user_id = u.id "
           "WHERE s.token = ?;");
  if (!sel.valid()) return std::nullopt;

  sel.bind_text(1, token);
  if (sqlite3_step(sel) != SQLITE_ROW) {
    return std::nullopt;
  }

  UserRecord user;
  user.id = sel.column_text(0);
  user.username = sel.column_text(1);
  user.password_hash = sel.column_text(2);
  user.salt = sel.column_text(3);
  user.created_at = sel.column_text(4);
  return user;
}

void ChatDatabase::Logout(const std::string& token) {
  if (token.empty()) return;
  std::lock_guard lock(mutex_);
  if (!db_) return;

  Stmt del(db_, "DELETE FROM sessions WHERE token = ?;");
  if (del.valid()) {
    del.bind_text(1, token);
    sqlite3_step(del);
  }
}

std::vector<ConversationRecord> ChatDatabase::ListConversations(
    const std::string& user_id) {
  std::vector<ConversationRecord> results;
  std::lock_guard lock(mutex_);
  if (!db_) return results;

  Stmt sel(db_,
           "SELECT c.id, c.user_id, c.title, c.model, c.system_prompt, c.working_directory, c.created_at, c.updated_at, "
           "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) "
           "FROM conversations c WHERE c.user_id = ? "
           "ORDER BY c.updated_at DESC;");
  if (!sel.valid()) return results;

  sel.bind_text(1, user_id);
  while (sqlite3_step(sel) == SQLITE_ROW) {
    ConversationRecord conv;
    conv.id = sel.column_text(0);
    conv.user_id = sel.column_text(1);
    conv.title = sel.column_text(2);
    conv.model = sel.column_text(3);
    conv.system_prompt = sel.column_text(4);
    conv.working_directory = sel.column_text(5);
    conv.created_at = sel.column_text(6);
    conv.updated_at = sel.column_text(7);
    const int count = sel.column_int(8);
    if (conv.working_directory.empty()) {
      conv.working_directory = GetConversationDefaultWorkingDir(conv.id, conv.title).string();
    }
    conv.messages.resize(static_cast<std::size_t>(count > 0 ? count : 0));
    results.push_back(std::move(conv));
  }

  return results;
}

std::optional<ConversationRecord> ChatDatabase::GetConversation(
    const std::string& user_id, const std::string& conv_id) {
  std::lock_guard lock(mutex_);
  if (!db_) return std::nullopt;

  ConversationRecord conv;
  {
    Stmt sel(db_,
             "SELECT id, user_id, title, model, system_prompt, working_directory, created_at, updated_at "
             "FROM conversations WHERE id = ? AND user_id = ?;");
    if (!sel.valid()) return std::nullopt;

    sel.bind_text(1, conv_id);
    sel.bind_text(2, user_id);
    if (sqlite3_step(sel) != SQLITE_ROW) {
      return std::nullopt;
    }

    conv.id = sel.column_text(0);
    conv.user_id = sel.column_text(1);
    conv.title = sel.column_text(2);
    conv.model = sel.column_text(3);
    conv.system_prompt = sel.column_text(4);
    conv.working_directory = sel.column_text(5);
    conv.created_at = sel.column_text(6);
    conv.updated_at = sel.column_text(7);
    if (conv.working_directory.empty()) {
      conv.working_directory = GetConversationDefaultWorkingDir(conv.id, conv.title).string();
    }
  }

  {
    Stmt sel_msg(db_,
                 "SELECT id, role, content, reasoning_content, created_at "
                 "FROM messages WHERE conversation_id = ? "
                 "ORDER BY order_idx ASC, rowid ASC;");
    if (sel_msg.valid()) {
      sel_msg.bind_text(1, conv_id);
      while (sqlite3_step(sel_msg) == SQLITE_ROW) {
        ChatMessageRecord msg;
        msg.id = sel_msg.column_text(0);
        msg.role = sel_msg.column_text(1);
        msg.content = sel_msg.column_text(2);
        msg.reasoning_content = sel_msg.column_text(3);
        msg.created_at = sel_msg.column_text(4);
        conv.messages.push_back(std::move(msg));
      }
    }
  }

  return conv;
}

ConversationRecord ChatDatabase::CreateConversation(
    const std::string& user_id, const std::string& title,
    const std::string& model, const std::string& system_prompt,
    const std::string& working_directory) {
  std::lock_guard lock(mutex_);

  ConversationRecord conv;
  conv.id = "conv_" + RandomHex(8);
  conv.user_id = user_id;
  conv.title = title.empty() ? "New Chat" : title;
  conv.model = model;
  conv.system_prompt = system_prompt;
  conv.created_at = CurrentIsoTime();
  conv.updated_at = conv.created_at;
  if (!working_directory.empty()) {
    conv.working_directory = working_directory;
  } else {
    conv.working_directory = GetConversationDefaultWorkingDir(conv.id, conv.title).string();
  }
  std::error_code ec;
  std::filesystem::create_directories(conv.working_directory, ec);

  if (db_) {
    Stmt ins(db_,
             "INSERT INTO conversations (id, user_id, title, model, system_prompt, working_directory, created_at, updated_at) "
             "VALUES (?, ?, ?, ?, ?, ?, ?, ?);");
    if (ins.valid()) {
      ins.bind_text(1, conv.id);
      ins.bind_text(2, conv.user_id);
      ins.bind_text(3, conv.title);
      ins.bind_text(4, conv.model);
      ins.bind_text(5, conv.system_prompt);
      ins.bind_text(6, conv.working_directory);
      ins.bind_text(7, conv.created_at);
      ins.bind_text(8, conv.updated_at);
      sqlite3_step(ins);
    }
  }

  return conv;
}

bool ChatDatabase::UpdateConversationTitle(const std::string& user_id,
                                          const std::string& conv_id,
                                          const std::string& new_title) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  std::string old_title;
  std::string curr_working_dir;
  {
    Stmt sel(db_, "SELECT title, working_directory FROM conversations WHERE id = ? AND user_id = ?;");
    if (sel.valid()) {
      sel.bind_text(1, conv_id);
      sel.bind_text(2, user_id);
      if (sqlite3_step(sel) == SQLITE_ROW) {
        old_title = sel.column_text(0);
        curr_working_dir = sel.column_text(1);
      }
    }
  }

  const std::string now = CurrentIsoTime();
  std::string new_working_dir = curr_working_dir;
  std::string old_default = GetConversationDefaultWorkingDir(conv_id, old_title).string();
  if (curr_working_dir.empty() || curr_working_dir == old_default) {
    new_working_dir = GetConversationDefaultWorkingDir(conv_id, new_title).string();
    std::error_code ec;
    std::filesystem::create_directories(new_working_dir, ec);
  }

  Stmt upd(db_,
           "UPDATE conversations SET title = ?, working_directory = ?, updated_at = ? WHERE id = ? AND user_id = ?;");
  if (!upd.valid()) return false;

  upd.bind_text(1, new_title);
  upd.bind_text(2, new_working_dir);
  upd.bind_text(3, now);
  upd.bind_text(4, conv_id);
  upd.bind_text(5, user_id);
  return sqlite3_step(upd) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool ChatDatabase::UpdateConversationWorkingDir(const std::string& user_id,
                                                const std::string& conv_id,
                                                const std::string& working_directory) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  std::string target_dir = working_directory;
  if (target_dir.empty() || target_dir == "default") {
    std::string title;
    Stmt sel(db_, "SELECT title FROM conversations WHERE id = ? AND user_id = ?;");
    if (sel.valid()) {
      sel.bind_text(1, conv_id);
      sel.bind_text(2, user_id);
      if (sqlite3_step(sel) == SQLITE_ROW) {
        title = sel.column_text(0);
      }
    }
    target_dir = GetConversationDefaultWorkingDir(conv_id, title).string();
  }

  std::error_code ec;
  std::filesystem::create_directories(target_dir, ec);

  const std::string now = CurrentIsoTime();
  Stmt upd(db_,
           "UPDATE conversations SET working_directory = ?, updated_at = ? WHERE id = ? AND user_id = ?;");
  if (!upd.valid()) return false;

  upd.bind_text(1, target_dir);
  upd.bind_text(2, now);
  upd.bind_text(3, conv_id);
  upd.bind_text(4, user_id);
  return sqlite3_step(upd) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool ChatDatabase::DeleteConversation(const std::string& user_id,
                                     const std::string& conv_id) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  // First verify ownership
  {
    Stmt check(db_, "SELECT id FROM conversations WHERE id = ? AND user_id = ?;");
    if (!check.valid()) return false;
    check.bind_text(1, conv_id);
    check.bind_text(2, user_id);
    if (sqlite3_step(check) != SQLITE_ROW) {
      return false;
    }
  }

  // Delete messages
  {
    Stmt del_msg(db_, "DELETE FROM messages WHERE conversation_id = ?;");
    if (del_msg.valid()) {
      del_msg.bind_text(1, conv_id);
      sqlite3_step(del_msg);
    }
  }

  // Delete conversation
  Stmt del_conv(db_, "DELETE FROM conversations WHERE id = ? AND user_id = ?;");
  if (!del_conv.valid()) return false;
  del_conv.bind_text(1, conv_id);
  del_conv.bind_text(2, user_id);
  return sqlite3_step(del_conv) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool ChatDatabase::AddMessage(const std::string& user_id,
                              const std::string& conv_id,
                              const ChatMessageRecord& msg) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  // Verify conversation ownership
  {
    Stmt check(db_, "SELECT id FROM conversations WHERE id = ? AND user_id = ?;");
    if (!check.valid()) return false;
    check.bind_text(1, conv_id);
    check.bind_text(2, user_id);
    if (sqlite3_step(check) != SQLITE_ROW) {
      return false;
    }
  }

  const std::string msg_id = msg.id.empty() ? ("msg_" + RandomHex(8)) : msg.id;
  const std::string created = msg.created_at.empty() ? CurrentIsoTime() : msg.created_at;

  Stmt ins(db_,
           "INSERT INTO messages (id, conversation_id, user_id, role, content, reasoning_content, created_at, order_idx) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, "
           "COALESCE((SELECT MAX(order_idx) + 1 FROM messages WHERE conversation_id = ?), 1));");
  if (!ins.valid()) return false;

  ins.bind_text(1, msg_id);
  ins.bind_text(2, conv_id);
  ins.bind_text(3, user_id);
  ins.bind_text(4, msg.role);
  ins.bind_text(5, msg.content);
  ins.bind_text(6, msg.reasoning_content);
  ins.bind_text(7, created);
  ins.bind_text(8, conv_id);

  if (sqlite3_step(ins) != SQLITE_DONE) {
    return false;
  }

  // Update conversation updated_at timestamp
  Stmt upd(db_, "UPDATE conversations SET updated_at = ? WHERE id = ?;");
  if (upd.valid()) {
    upd.bind_text(1, created);
    upd.bind_text(2, conv_id);
    sqlite3_step(upd);
  }

  return true;
}

UserSettingsRecord ChatDatabase::GetSettings(const std::string& user_id) {
  std::lock_guard lock(mutex_);
  UserSettingsRecord st;
  st.user_id = user_id;
  if (!db_) return st;

  Stmt sel(db_,
           "SELECT temperature, top_p, max_tokens, system_prompt, working_directory FROM user_settings "
           "WHERE user_id = ?;");
  if (!sel.valid()) return st;

  sel.bind_text(1, user_id);
  if (sqlite3_step(sel) == SQLITE_ROW) {
    st.temperature = sel.column_double(0);
    st.top_p = sel.column_double(1);
    st.max_tokens = sel.column_int(2);
    st.system_prompt = sel.column_text(3);
    st.working_directory = sel.column_text(4);
  }

  return st;
}

void ChatDatabase::SaveSettings(const UserSettingsRecord& settings) {
  std::lock_guard lock(mutex_);
  if (!db_) return;

  Stmt ins(db_,
           "INSERT INTO user_settings (user_id, temperature, top_p, max_tokens, system_prompt, working_directory) "
           "VALUES (?, ?, ?, ?, ?, ?) "
           "ON CONFLICT(user_id) DO UPDATE SET "
           "temperature = excluded.temperature, "
           "top_p = excluded.top_p, "
           "max_tokens = excluded.max_tokens, "
           "system_prompt = excluded.system_prompt, "
           "working_directory = excluded.working_directory;");
  if (!ins.valid()) return;

  ins.bind_text(1, settings.user_id);
  ins.bind_double(2, settings.temperature);
  ins.bind_double(3, settings.top_p);
  ins.bind_int(4, settings.max_tokens);
  ins.bind_text(5, settings.system_prompt);
  ins.bind_text(6, settings.working_directory);
  sqlite3_step(ins);
}

std::vector<McpServerRecord> ChatDatabase::ListMcpServers(const std::string& user_id) {
  std::vector<McpServerRecord> list;
  std::lock_guard lock(mutex_);
  if (!db_) return list;

  Stmt sel(db_,
           "SELECT id, user_id, name, url, transport, headers_json, tools_cache_json, enabled, created_at "
           "FROM mcp_servers WHERE user_id = ? ORDER BY created_at ASC;");
  if (!sel.valid()) return list;

  sel.bind_text(1, user_id);
  while (sqlite3_step(sel) == SQLITE_ROW) {
    McpServerRecord rec;
    rec.id = sel.column_text(0);
    rec.user_id = sel.column_text(1);
    rec.name = sel.column_text(2);
    rec.url = sel.column_text(3);
    rec.transport = sel.column_text(4);
    rec.headers_json = sel.column_text(5);
    rec.tools_cache_json = sel.column_text(6);
    rec.enabled = sel.column_int(7) != 0;
    rec.created_at = sel.column_text(8);
    list.push_back(std::move(rec));
  }
  return list;
}

McpServerRecord ChatDatabase::AddMcpServer(const std::string& user_id,
                                          const std::string& name,
                                          const std::string& url,
                                          const std::string& transport,
                                          const std::string& headers_json) {
  std::lock_guard lock(mutex_);
  McpServerRecord rec;
  rec.id = "mcp_" + RandomHex(8);
  rec.user_id = user_id;
  rec.name = name.empty() ? "MCP Server" : name;
  rec.url = url;
  rec.transport = transport.empty() ? "sse" : transport;
  rec.headers_json = headers_json.empty() ? "{}" : headers_json;
  rec.tools_cache_json = "[]";
  rec.enabled = true;
  rec.created_at = CurrentIsoTime();

  if (db_) {
    Stmt ins(db_,
             "INSERT INTO mcp_servers (id, user_id, name, url, transport, headers_json, tools_cache_json, enabled, created_at) "
             "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);");
    if (ins.valid()) {
      ins.bind_text(1, rec.id);
      ins.bind_text(2, rec.user_id);
      ins.bind_text(3, rec.name);
      ins.bind_text(4, rec.url);
      ins.bind_text(5, rec.transport);
      ins.bind_text(6, rec.headers_json);
      ins.bind_text(7, rec.tools_cache_json);
      ins.bind_int(8, rec.enabled ? 1 : 0);
      ins.bind_text(9, rec.created_at);
      sqlite3_step(ins);
    }
  }
  return rec;
}

bool ChatDatabase::UpdateMcpServer(const std::string& user_id,
                                  const McpServerRecord& server) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  Stmt upd(db_,
           "UPDATE mcp_servers SET name = ?, url = ?, transport = ?, headers_json = ?, "
           "tools_cache_json = ?, enabled = ? WHERE id = ? AND user_id = ?;");
  if (!upd.valid()) return false;

  upd.bind_text(1, server.name);
  upd.bind_text(2, server.url);
  upd.bind_text(3, server.transport);
  upd.bind_text(4, server.headers_json);
  upd.bind_text(5, server.tools_cache_json);
  upd.bind_int(6, server.enabled ? 1 : 0);
  upd.bind_text(7, server.id);
  upd.bind_text(8, user_id);
  return sqlite3_step(upd) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool ChatDatabase::DeleteMcpServer(const std::string& user_id,
                                  const std::string& server_id) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  Stmt del(db_, "DELETE FROM mcp_servers WHERE id = ? AND user_id = ?;");
  if (!del.valid()) return false;
  del.bind_text(1, server_id);
  del.bind_text(2, user_id);
  return sqlite3_step(del) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

std::vector<CustomToolRecord> ChatDatabase::ListCustomTools(const std::string& user_id) {
  std::vector<CustomToolRecord> list;
  std::lock_guard lock(mutex_);
  if (!db_) return list;

  Stmt sel(db_,
           "SELECT id, user_id, name, description, schema_json, handler_url, enabled, created_at "
           "FROM custom_tools WHERE user_id = ? ORDER BY created_at ASC;");
  if (!sel.valid()) return list;

  sel.bind_text(1, user_id);
  while (sqlite3_step(sel) == SQLITE_ROW) {
    CustomToolRecord rec;
    rec.id = sel.column_text(0);
    rec.user_id = sel.column_text(1);
    rec.name = sel.column_text(2);
    rec.description = sel.column_text(3);
    rec.schema_json = sel.column_text(4);
    rec.handler_url = sel.column_text(5);
    rec.enabled = sel.column_int(6) != 0;
    rec.created_at = sel.column_text(7);
    list.push_back(std::move(rec));
  }
  return list;
}

CustomToolRecord ChatDatabase::AddCustomTool(const std::string& user_id,
                                           const std::string& name,
                                           const std::string& description,
                                           const std::string& schema_json,
                                           const std::string& handler_url) {
  std::lock_guard lock(mutex_);
  CustomToolRecord rec;
  rec.id = "tool_" + RandomHex(8);
  rec.user_id = user_id;
  rec.name = name;
  rec.description = description;
  rec.schema_json = schema_json.empty() ? "{}" : schema_json;
  rec.handler_url = handler_url;
  rec.enabled = true;
  rec.created_at = CurrentIsoTime();

  if (db_) {
    Stmt ins(db_,
             "INSERT INTO custom_tools (id, user_id, name, description, schema_json, handler_url, enabled, created_at) "
             "VALUES (?, ?, ?, ?, ?, ?, ?, ?);");
    if (ins.valid()) {
      ins.bind_text(1, rec.id);
      ins.bind_text(2, rec.user_id);
      ins.bind_text(3, rec.name);
      ins.bind_text(4, rec.description);
      ins.bind_text(5, rec.schema_json);
      ins.bind_text(6, rec.handler_url);
      ins.bind_int(7, rec.enabled ? 1 : 0);
      ins.bind_text(8, rec.created_at);
      sqlite3_step(ins);
    }
  }
  return rec;
}

bool ChatDatabase::UpdateCustomTool(const std::string& user_id,
                                   const CustomToolRecord& tool) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  Stmt upd(db_,
           "UPDATE custom_tools SET name = ?, description = ?, schema_json = ?, "
           "handler_url = ?, enabled = ? WHERE id = ? AND user_id = ?;");
  if (!upd.valid()) return false;

  upd.bind_text(1, tool.name);
  upd.bind_text(2, tool.description);
  upd.bind_text(3, tool.schema_json);
  upd.bind_text(4, tool.handler_url);
  upd.bind_int(5, tool.enabled ? 1 : 0);
  upd.bind_text(6, tool.id);
  upd.bind_text(7, user_id);
  return sqlite3_step(upd) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool ChatDatabase::DeleteCustomTool(const std::string& user_id,
                                   const std::string& tool_id) {
  std::lock_guard lock(mutex_);
  if (!db_) return false;

  Stmt del(db_, "DELETE FROM custom_tools WHERE id = ? AND user_id = ?;");
  if (!del.valid()) return false;
  del.bind_text(1, tool_id);
  del.bind_text(2, user_id);
  return sqlite3_step(del) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

std::string ChatDatabase::GetInbuiltToolsConfig(const std::string& user_id) {
  std::lock_guard lock(mutex_);
  if (!db_) return "{}";

  Stmt sel(db_, "SELECT inbuilt_tools_json FROM user_tools_config WHERE user_id = ?;");
  if (!sel.valid()) return "{}";

  sel.bind_text(1, user_id);
  if (sqlite3_step(sel) == SQLITE_ROW) {
    return sel.column_text(0);
  }
  return "{}";
}

void ChatDatabase::SaveInbuiltToolsConfig(const std::string& user_id,
                                         const std::string& config_json) {
  std::lock_guard lock(mutex_);
  if (!db_) return;

  Stmt ins(db_,
           "INSERT INTO user_tools_config (user_id, inbuilt_tools_json) "
           "VALUES (?, ?) "
           "ON CONFLICT(user_id) DO UPDATE SET "
           "inbuilt_tools_json = excluded.inbuilt_tools_json;");
  if (!ins.valid()) return;

  ins.bind_text(1, user_id);
  ins.bind_text(2, config_json);
  sqlite3_step(ins);
}

}  // namespace gufo::server
