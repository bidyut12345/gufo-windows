#ifndef GUFO_SERVER_CHAT_PORTAL_DB_HPP_
#define GUFO_SERVER_CHAT_PORTAL_DB_HPP_

#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "src/core/json.hpp"

struct sqlite3;

namespace gufo::server {

struct ChatMessageRecord {
  std::string id;
  std::string role;             // "user", "assistant", "system", "tool"
  std::string content;
  std::string reasoning_content;
  std::string created_at;
};

struct ConversationRecord {
  std::string id;
  std::string user_id;
  std::string title;
  std::string model;
  std::string system_prompt;
  std::string working_directory;
  std::string created_at;
  std::string updated_at;
  std::vector<ChatMessageRecord> messages;
};

struct UserRecord {
  std::string id;
  std::string username;
  std::string password_hash;
  std::string salt;
  std::string created_at;
};

struct SessionRecord {
  std::string token;
  std::string user_id;
  std::string created_at;
  std::string expires_at;
};

struct UserSettingsRecord {
  std::string user_id;
  double temperature{0.6};
  double top_p{0.95};
  int max_tokens{4096};
  std::string system_prompt{
      "You are Gufo, an ultra-fast, helpful AI assistant running locally on AMD Strix Halo gfx1151."};
  std::string working_directory;
};

struct McpServerRecord {
  std::string id;
  std::string user_id;
  std::string name;
  std::string url;
  std::string transport{"sse"};  // "sse" or "http"
  std::string headers_json{"{}"};
  std::string tools_cache_json{"[]"};
  bool enabled{true};
  std::string created_at;
};

struct CustomToolRecord {
  std::string id;
  std::string user_id;
  std::string name;
  std::string description;
  std::string schema_json;
  std::string handler_url;
  bool enabled{true};
  std::string created_at;
};

std::filesystem::path GetExecutableDirectory();
std::string SanitizeFolderName(const std::string& raw);
std::filesystem::path GetConversationDefaultWorkingDir(const std::string& conv_id, const std::string& title);

class ChatDatabase {
 public:
  static ChatDatabase& Instance();

  explicit ChatDatabase(std::filesystem::path db_path = {});
  ~ChatDatabase();

  ChatDatabase(const ChatDatabase&) = delete;
  ChatDatabase& operator=(const ChatDatabase&) = delete;
  ChatDatabase(ChatDatabase&&) = delete;
  ChatDatabase& operator=(ChatDatabase&&) = delete;

  struct AuthResult {
    bool success{false};
    std::string error;
    std::string token;
    UserRecord user;
  };

  AuthResult Register(const std::string& username, const std::string& password);
  AuthResult Login(const std::string& username, const std::string& password);
  std::optional<UserRecord> AuthenticateToken(const std::string& token);
  void Logout(const std::string& token);

  std::vector<ConversationRecord> ListConversations(const std::string& user_id);
  std::optional<ConversationRecord> GetConversation(const std::string& user_id,
                                                    const std::string& conv_id);
  ConversationRecord CreateConversation(const std::string& user_id,
                                        const std::string& title,
                                        const std::string& model,
                                        const std::string& system_prompt,
                                        const std::string& working_directory = "");
  bool UpdateConversationTitle(const std::string& user_id,
                               const std::string& conv_id,
                               const std::string& new_title);
  bool UpdateConversationWorkingDir(const std::string& user_id,
                                    const std::string& conv_id,
                                    const std::string& working_directory);
  bool DeleteConversation(const std::string& user_id,
                          const std::string& conv_id);
  bool AddMessage(const std::string& user_id, const std::string& conv_id,
                  const ChatMessageRecord& msg);

  UserSettingsRecord GetSettings(const std::string& user_id);
  void SaveSettings(const UserSettingsRecord& settings);

  // MCP Servers
  std::vector<McpServerRecord> ListMcpServers(const std::string& user_id);
  McpServerRecord AddMcpServer(const std::string& user_id,
                               const std::string& name,
                               const std::string& url,
                               const std::string& transport,
                               const std::string& headers_json);
  bool UpdateMcpServer(const std::string& user_id,
                       const McpServerRecord& server);
  bool DeleteMcpServer(const std::string& user_id,
                       const std::string& server_id);

  // Custom Tools
  std::vector<CustomToolRecord> ListCustomTools(const std::string& user_id);
  CustomToolRecord AddCustomTool(const std::string& user_id,
                                const std::string& name,
                                const std::string& description,
                                const std::string& schema_json,
                                const std::string& handler_url);
  bool UpdateCustomTool(const std::string& user_id,
                        const CustomToolRecord& tool);
  bool DeleteCustomTool(const std::string& user_id,
                        const std::string& tool_id);

  // Inbuilt tools enablement configuration (JSON object of tool names to bool)
  std::string GetInbuiltToolsConfig(const std::string& user_id);
  void SaveInbuiltToolsConfig(const std::string& user_id,
                              const std::string& config_json);

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  void InitSchema();
  static std::filesystem::path DefaultDbPath();

  std::filesystem::path path_;
  std::mutex mutex_;
  sqlite3* db_{nullptr};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CHAT_PORTAL_DB_HPP_
