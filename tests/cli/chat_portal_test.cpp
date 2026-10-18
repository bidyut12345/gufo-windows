#include <cassert>
#include <filesystem>
#include <iostream>

#include "src/cli/serve/chat_portal_db.hpp"
#include "src/cli/serve/chat_portal_service.hpp"
#include "src/cli/serve/chat_portal_tools.hpp"

namespace {

void TestChatDatabasePersistence() {
  const auto test_db_path =
      std::filesystem::temp_directory_path() / "gufo_chat_portal_test.db";
  std::error_code ec;
  std::filesystem::remove(test_db_path, ec);
  std::filesystem::remove(test_db_path.string() + "-wal", ec);
  std::filesystem::remove(test_db_path.string() + "-shm", ec);

  // Scope 1: Write data
  {
    gufo::server::ChatDatabase db(test_db_path);

    // Register user
    const auto reg_res = db.Register("alice", "secret123");
    assert(reg_res.success);
    assert(!reg_res.token.empty());
    assert(reg_res.user.username == "alice");

    // Duplicate user should fail
    const auto dup_res = db.Register("Alice", "otherpass");
    assert(!dup_res.success);

    // Login test
    const auto login_fail = db.Login("alice", "wrongpass");
    assert(!login_fail.success);

    const auto login_ok = db.Login("alice", "secret123");
    assert(login_ok.success);
    assert(!login_ok.token.empty());

    // Validate token
    const auto auth_user = db.AuthenticateToken(login_ok.token);
    assert(auth_user.has_value());
    assert(auth_user->username == "alice");

    // Create conversation
    const auto conv = db.CreateConversation(
        auth_user->id, "Test Discussion", "Qwen3.8-Flash-Next", "System prompt");
    assert(!conv.id.empty());
    assert(conv.title == "Test Discussion");

    // Add message
    gufo::server::ChatMessageRecord msg;
    msg.role = "user";
    msg.content = "Hello Gufo!";
    assert(db.AddMessage(auth_user->id, conv.id, msg));

    gufo::server::ChatMessageRecord asst_msg;
    asst_msg.role = "assistant";
    asst_msg.content = "Hello Alice!";
    asst_msg.reasoning_content = "Thinking about greeting...";
    assert(db.AddMessage(auth_user->id, conv.id, asst_msg));

    // Update title
    assert(db.UpdateConversationTitle(auth_user->id, conv.id, "Renamed Title"));

    // Settings
    gufo::server::UserSettingsRecord st;
    st.user_id = auth_user->id;
    st.temperature = 0.75;
    st.top_p = 0.90;
    st.max_tokens = 2048;
    st.system_prompt = "Custom prompt";
    db.SaveSettings(st);
  }

  // Scope 2: Re-open database from disk and verify persistence
  {
    gufo::server::ChatDatabase db(test_db_path);

    const auto login_ok = db.Login("alice", "secret123");
    assert(login_ok.success);
    const std::string user_id = login_ok.user.id;

    const auto list = db.ListConversations(user_id);
    assert(list.size() == 1);
    assert(list[0].title == "Renamed Title");

    const auto conv = db.GetConversation(user_id, list[0].id);
    assert(conv.has_value());
    assert(conv->messages.size() == 2);
    assert(conv->messages[0].role == "user");
    assert(conv->messages[0].content == "Hello Gufo!");
    assert(conv->messages[1].role == "assistant");
    assert(conv->messages[1].content == "Hello Alice!");
    assert(conv->messages[1].reasoning_content == "Thinking about greeting...");

    const auto st = db.GetSettings(user_id);
    assert(st.temperature == 0.75);
    assert(st.top_p == 0.90);
    assert(st.max_tokens == 2048);
    assert(st.system_prompt == "Custom prompt");

    // Delete conversation
    assert(db.DeleteConversation(user_id, conv->id));
    assert(db.ListConversations(user_id).empty());
  }

  std::filesystem::remove(test_db_path, ec);
  std::filesystem::remove(test_db_path.string() + "-wal", ec);
  std::filesystem::remove(test_db_path.string() + "-shm", ec);
  std::cout << "[PASS] TestChatDatabasePersistence\n";
}

void TestChatPortalService() {
  const auto test_db_path =
      std::filesystem::temp_directory_path() / "gufo_chat_service_test.db";
  std::error_code ec;
  std::filesystem::remove(test_db_path, ec);
  std::filesystem::remove(test_db_path.string() + "-wal", ec);
  std::filesystem::remove(test_db_path.string() + "-shm", ec);

  gufo::server::ChatDatabase db(test_db_path);

  // Check portal route detection
  gufo::server::HttpRequest req_home;
  req_home.method = "GET";
  req_home.path = "/";
  assert(gufo::server::ChatPortalService::IsPortalRoute(req_home));

  const auto resp_home =
      gufo::server::ChatPortalService::Handle(req_home, db);
  assert(resp_home.status == 200);
  assert(resp_home.body.find("Gufo") != std::string::npos);
  assert(resp_home.body.find("<!DOCTYPE html>") != std::string::npos);

  // Register via API
  gufo::server::HttpRequest req_reg;
  req_reg.method = "POST";
  req_reg.path = "/api/auth/register";
  req_reg.body = R"({"username":"bob","password":"password123"})";
  const auto resp_reg =
      gufo::server::ChatPortalService::Handle(req_reg, db);
  assert(resp_reg.status == 200);
  assert(resp_reg.body.find("gufotok_") != std::string::npos);

  // Extract token from register response
  const auto parsed = gufo::json::parse(resp_reg.body);
  const std::string token = parsed.member_str("token");
  assert(!token.empty());

  // GET /api/auth/me
  gufo::server::HttpRequest req_me;
  req_me.method = "GET";
  req_me.path = "/api/auth/me";
  req_me.headers.emplace_back("Authorization", "Bearer " + token);
  const auto resp_me = gufo::server::ChatPortalService::Handle(req_me, db);
  assert(resp_me.status == 200);
  assert(resp_me.body.find("bob") != std::string::npos);

  // Unauthenticated request
  gufo::server::HttpRequest req_unauth;
  req_unauth.method = "GET";
  req_unauth.path = "/api/conversations";
  const auto resp_unauth =
      gufo::server::ChatPortalService::Handle(req_unauth, db);
  assert(resp_unauth.status == 401);

  // Authenticated list conversations (empty)
  gufo::server::HttpRequest req_list;
  req_list.method = "GET";
  req_list.path = "/api/conversations";
  req_list.headers.emplace_back("Authorization", "Bearer " + token);
  const auto resp_list =
      gufo::server::ChatPortalService::Handle(req_list, db);
  assert(resp_list.status == 200);
  assert(resp_list.body == "[]");

  // Create conversation
  gufo::server::HttpRequest req_create;
  req_create.method = "POST";
  req_create.path = "/api/conversations";
  req_create.headers.emplace_back("Authorization", "Bearer " + token);
  req_create.body =
      R"({"title":"Bob's Chat","model":"test-model","system_prompt":""})";
  const auto resp_create =
      gufo::server::ChatPortalService::Handle(req_create, db);
  assert(resp_create.status == 200);
  const auto conv_obj = gufo::json::parse(resp_create.body);
  const std::string conv_id = conv_obj.member_str("id");
  assert(!conv_id.empty());

  // Add message
  gufo::server::HttpRequest req_msg;
  req_msg.method = "POST";
  req_msg.path = "/api/conversations/" + conv_id + "/messages";
  req_msg.headers.emplace_back("Authorization", "Bearer " + token);
  req_msg.body = R"({"role":"user","content":"How fast is Strix Halo?"})";
  const auto resp_msg =
      gufo::server::ChatPortalService::Handle(req_msg, db);
  assert(resp_msg.status == 200);

  // Get conversation with message
  gufo::server::HttpRequest req_get;
  req_get.method = "GET";
  req_get.path = "/api/conversations/" + conv_id;
  req_get.headers.emplace_back("Authorization", "Bearer " + token);
  const auto resp_get =
      gufo::server::ChatPortalService::Handle(req_get, db);
  assert(resp_get.status == 200);
  assert(resp_get.body.find("How fast is Strix Halo?") != std::string::npos);

  std::filesystem::remove(test_db_path, ec);
  std::filesystem::remove(test_db_path.string() + "-wal", ec);
  std::filesystem::remove(test_db_path.string() + "-shm", ec);
  std::cout << "[PASS] TestChatPortalService\n";
}

void TestToolsAndMcp() {
  // 1. Inbuilt tool inventory
  const auto& tools = gufo::server::ToolEngine::GetInbuiltTools();
  assert(tools.size() == 9);
  assert(gufo::server::ToolEngine::FindInbuiltTool("get_datetime") != nullptr);
  assert(gufo::server::ToolEngine::FindInbuiltTool("read_file") != nullptr);
  assert(gufo::server::ToolEngine::FindInbuiltTool("write_file") != nullptr);
  assert(gufo::server::ToolEngine::FindInbuiltTool("exec_shell_command") != nullptr);

  // 2. Execute get_datetime
  const auto dt_res = gufo::server::ToolEngine::Execute("get_datetime", gufo::json::Value::object());
  assert(dt_res.success);
  assert(dt_res.output.find("datetime") != std::string::npos);

  // 3. Execute write_file, read_file, edit_file
  const auto temp_file = std::filesystem::temp_directory_path() / "gufo_tool_test.txt";
  std::error_code ec;
  std::filesystem::remove(temp_file, ec);

  gufo::json::Value write_args = gufo::json::Value::object();
  write_args["path"] = temp_file.string();
  write_args["content"] = "Line 1: Strix Halo\nLine 2: gfx1151\nLine 3: Fast LLM\n";
  const auto write_res = gufo::server::ToolEngine::Execute("write_file", write_args);
  assert(write_res.success);

  gufo::json::Value read_args = gufo::json::Value::object();
  read_args["path"] = temp_file.string();
  read_args["offset"] = 2;
  read_args["limit"] = 1;
  const auto read_res = gufo::server::ToolEngine::Execute("read_file", read_args);
  assert(read_res.success);
  assert(read_res.output.find("gfx1151") != std::string::npos);

  gufo::json::Value edit_args = gufo::json::Value::object();
  edit_args["path"] = temp_file.string();
  edit_args["target"] = "Fast LLM";
  edit_args["replacement"] = "Ultra Fast LLM";
  const auto edit_res = gufo::server::ToolEngine::Execute("edit_file", edit_args);
  assert(edit_res.success);

  std::filesystem::remove(temp_file, ec);

  // 4. Test MCP Servers and Custom Tools SQLite persistence
  const auto test_db_path =
      std::filesystem::temp_directory_path() / "gufo_tools_db_test.db";
  std::filesystem::remove(test_db_path, ec);
  std::filesystem::remove(test_db_path.string() + "-wal", ec);
  std::filesystem::remove(test_db_path.string() + "-shm", ec);

  {
    gufo::server::ChatDatabase db(test_db_path);
    const auto reg = db.Register("engineer", "pass123");
    assert(reg.success);

    // Add MCP server
    const auto srv = db.AddMcpServer(reg.user.id, "Local Time MCP", "http://localhost:8001/sse", "sse", "{}");
    assert(!srv.id.empty());
    assert(srv.name == "Local Time MCP");

    const auto srvs = db.ListMcpServers(reg.user.id);
    assert(srvs.size() == 1);
    assert(srvs[0].url == "http://localhost:8001/sse");

    // Add Custom Tool
    const auto ctool = db.AddCustomTool(reg.user.id, "calc_add", "Adds two numbers", R"({"type":"object"})", "");
    assert(!ctool.id.empty());
    assert(ctool.name == "calc_add");

    const auto ctools = db.ListCustomTools(reg.user.id);
    assert(ctools.size() == 1);
    assert(ctools[0].name == "calc_add");

    // Service route check
    gufo::server::HttpRequest req_tools;
    req_tools.method = "GET";
    req_tools.path = "/api/tools";
    req_tools.headers.emplace_back("Authorization", "Bearer " + reg.token);
    const auto resp_tools = gufo::server::ChatPortalService::Handle(req_tools, db);
    assert(resp_tools.status == 200);
    assert(resp_tools.body.find("get_datetime") != std::string::npos);
    assert(resp_tools.body.find("Local Time MCP") != std::string::npos);
    assert(resp_tools.body.find("calc_add") != std::string::npos);

    // Execute tool via service API
    gufo::server::HttpRequest req_exec;
    req_exec.method = "POST";
    req_exec.path = "/api/tools/execute";
    req_exec.headers.emplace_back("Authorization", "Bearer " + reg.token);
    req_exec.body = R"({"name":"get_datetime","arguments":{}})";
    const auto resp_exec = gufo::server::ChatPortalService::Handle(req_exec, db);
    assert(resp_exec.status == 200);
    assert(resp_exec.body.find("datetime") != std::string::npos);
  }

  std::filesystem::remove(test_db_path, ec);
  std::filesystem::remove(test_db_path.string() + "-wal", ec);
  std::filesystem::remove(test_db_path.string() + "-shm", ec);
  std::cout << "[PASS] TestToolsAndMcp\n";
}

}  // namespace

int main() {
  TestChatDatabasePersistence();
  TestChatPortalService();
  TestToolsAndMcp();
  std::cout << "All chat portal and MCP tools tests passed successfully.\n";
  return 0;
}
