#include "src/cli/serve/chat_portal_service.hpp"

#include <string_view>

#include "src/cli/serve/chat_portal_html.hpp"
#include "src/cli/serve/chat_portal_tools.hpp"
#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {

std::string ToLower(std::string_view s) {
  std::string out(s);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

HttpResponse JsonResponse(int status, const std::string& reason,
                         const std::string& body) {
  HttpResponse resp;
  resp.status = status;
  resp.reason = reason;
  resp.body = body;
  resp.headers.emplace_back("Content-Type", "application/json; charset=utf-8");
  resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
  resp.headers.emplace_back("Access-Control-Allow-Headers", "*");
  resp.headers.emplace_back("Access-Control-Allow-Methods",
                           "GET, POST, PATCH, DELETE, OPTIONS");
  return resp;
}

HttpResponse JsonOk(const json::Value& val) {
  return JsonResponse(200, "OK", val.dump());
}

HttpResponse JsonErr(int status, const std::string& reason,
                     const std::string& message) {
  json::Value obj = json::Value::object();
  obj["error"] = message;
  return JsonResponse(status, reason, obj.dump());
}

}  // namespace

bool ChatPortalService::IsPortalRoute(const HttpRequest& req) {
  return req.path == "/" || req.path == "/index.html" ||
         req.path == "/favicon.ico" || req.path.starts_with("/api/");
}

std::string ChatPortalService::ExtractBearerToken(const HttpRequest& req) {
  for (const auto& [name, value] : req.headers) {
    if (ToLower(name) == "authorization") {
      std::string_view credential = value;
      const auto space = credential.find(' ');
      if (space != std::string_view::npos &&
          ToLower(credential.substr(0, space)) == "bearer") {
        credential.remove_prefix(space + 1);
        while (credential.starts_with(' ')) {
          credential.remove_prefix(1);
        }
        return std::string(credential);
      }
    }
  }
  return "";
}

HttpResponse ChatPortalService::Handle(const HttpRequest& req,
                                       ChatDatabase& db) {
  if (req.method == "OPTIONS") {
    HttpResponse resp;
    resp.status = 204;
    resp.reason = "No Content";
    resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
    resp.headers.emplace_back("Access-Control-Allow-Headers", "*");
    resp.headers.emplace_back("Access-Control-Allow-Methods",
                             "GET, POST, PATCH, DELETE, OPTIONS");
    return resp;
  }

  // Web portal single page application
  if (req.method == "GET" && (req.path == "/" || req.path == "/index.html")) {
    HttpResponse resp;
    resp.status = 200;
    resp.reason = "OK";
    resp.body = std::string(kChatPortalHtml);
    resp.headers.emplace_back("Content-Type", "text/html; charset=utf-8");
    resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
    return resp;
  }

  if (req.method == "GET" && req.path == "/favicon.ico") {
    HttpResponse resp;
    resp.status = 200;
    resp.reason = "OK";
    resp.body =
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        "<text y='.9em' font-size='90'>\xF0\x9F\xA6\x89</text></svg>";
    resp.headers.emplace_back("Content-Type", "image/svg+xml");
    resp.headers.emplace_back("Cache-Control", "public, max-age=86400");
    return resp;
  }

  // --- Auth APIs ---
  if (req.path == "/api/auth/register" && req.method == "POST") {
    try {
      const auto body = json::parse(req.body);
      const std::string username = body.member_str("username");
      const std::string password = body.member_str("password");
      const auto res = db.Register(username, password);
      if (!res.success) {
        return JsonErr(400, "Bad Request", res.error);
      }
      json::Value out = json::Value::object();
      out["token"] = res.token;
      json::Value user_obj = json::Value::object();
      user_obj["id"] = res.user.id;
      user_obj["username"] = res.user.username;
      out["user"] = std::move(user_obj);
      return JsonOk(out);
    } catch (...) {
      return JsonErr(400, "Bad Request", "Invalid JSON payload");
    }
  }

  if (req.path == "/api/auth/login" && req.method == "POST") {
    try {
      const auto body = json::parse(req.body);
      const std::string username = body.member_str("username");
      const std::string password = body.member_str("password");
      const auto res = db.Login(username, password);
      if (!res.success) {
        return JsonErr(401, "Unauthorized", res.error);
      }
      json::Value out = json::Value::object();
      out["token"] = res.token;
      json::Value user_obj = json::Value::object();
      user_obj["id"] = res.user.id;
      user_obj["username"] = res.user.username;
      out["user"] = std::move(user_obj);
      return JsonOk(out);
    } catch (...) {
      return JsonErr(400, "Bad Request", "Invalid JSON payload");
    }
  }

  if (req.path == "/api/auth/logout" && req.method == "POST") {
    const auto token = ExtractBearerToken(req);
    db.Logout(token);
    json::Value out = json::Value::object();
    out["status"] = "ok";
    return JsonOk(out);
  }

  // Authenticate token or fallback to guest for portal APIs
  const auto token = ExtractBearerToken(req);
  auto user = db.AuthenticateToken(token);
  if (!user.has_value()) {
    if (token == "guest") {
      user = UserRecord{.id = "u_guest",
                        .username = "Guest",
                        .password_hash = "",
                        .salt = "",
                        .created_at = ""};
    } else {
      return JsonErr(401, "Unauthorized", "Authentication required");
    }
  }

  if (req.path == "/api/auth/me" && req.method == "GET") {
    json::Value out = json::Value::object();
    json::Value user_obj = json::Value::object();
    user_obj["id"] = user->id;
    user_obj["username"] = user->username;
    out["user"] = std::move(user_obj);

    const auto st = db.GetSettings(user->id);
    json::Value st_obj = json::Value::object();
    st_obj["temperature"] = st.temperature;
    st_obj["top_p"] = st.top_p;
    st_obj["max_tokens"] = st.max_tokens;
    st_obj["system_prompt"] = st.system_prompt;
    st_obj["working_directory"] = st.working_directory;
    out["settings"] = std::move(st_obj);
    return JsonOk(out);
  }

  // --- Conversations APIs ---
  if (req.path == "/api/conversations" && req.method == "GET") {
    const auto list = db.ListConversations(user->id);
    json::Value arr = json::Value::array();
    for (const auto& c : list) {
      json::Value obj = json::Value::object();
      obj["id"] = c.id;
      obj["title"] = c.title;
      obj["model"] = c.model;
      obj["working_directory"] = c.working_directory;
      obj["updated_at"] = c.updated_at;
      obj["message_count"] = static_cast<int>(c.messages.size());
      arr.push_back(std::move(obj));
    }
    return JsonOk(arr);
  }

  if (req.path == "/api/conversations" && req.method == "POST") {
    try {
      const auto body = json::parse(req.body);
      const std::string title = body.member_str("title");
      const std::string model = body.member_str("model");
      const std::string system_prompt = body.member_str("system_prompt");
      const std::string working_dir = body.member_str("working_directory");
      const auto conv = db.CreateConversation(user->id, title, model, system_prompt, working_dir);

      json::Value obj = json::Value::object();
      obj["id"] = conv.id;
      obj["title"] = conv.title;
      obj["model"] = conv.model;
      obj["system_prompt"] = conv.system_prompt;
      obj["working_directory"] = conv.working_directory;
      obj["created_at"] = conv.created_at;
      obj["updated_at"] = conv.updated_at;
      obj["messages"] = json::Value::array();
      return JsonOk(obj);
    } catch (...) {
      return JsonErr(400, "Bad Request", "Invalid JSON payload");
    }
  }

  if (req.path.starts_with("/api/conversations/")) {
    const std::string rest = req.path.substr(std::string("/api/conversations/").size());
    const auto slash = rest.find('/');
    const std::string conv_id = slash == std::string::npos ? rest : rest.substr(0, slash);
    const std::string sub_route = slash == std::string::npos ? "" : rest.substr(slash);

    // POST /api/conversations/<id>/messages
    if (sub_route == "/messages" && req.method == "POST") {
      try {
        const auto body = json::parse(req.body);
        ChatMessageRecord msg;
        msg.role = body.member_str("role");
        msg.content = body.member_str("content");
        msg.reasoning_content = body.member_str("reasoning_content");
        if (db.AddMessage(user->id, conv_id, msg)) {
          json::Value out = json::Value::object();
          out["status"] = "ok";
          return JsonOk(out);
        }
        return JsonErr(404, "Not Found", "Conversation not found");
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }

    // PUT /api/conversations/<id>/messages (Replace all messages, used by context compaction)
    if (sub_route == "/messages" && req.method == "PUT") {
      try {
        const auto body = json::parse(req.body);
        std::vector<ChatMessageRecord> msgs;
        const auto* arr = body.is_array() ? &body : body.find("messages");
        if (arr && arr->is_array()) {
          for (const auto& item : arr->items()) {
            ChatMessageRecord msg;
            msg.id = item.member_str("id");
            msg.role = item.member_str("role");
            msg.content = item.member_str("content");
            msg.reasoning_content = item.member_str("reasoning_content");
            msgs.push_back(std::move(msg));
          }
        }
        if (db.ReplaceMessages(user->id, conv_id, msgs)) {
          json::Value out = json::Value::object();
          out["status"] = "ok";
          out["count"] = static_cast<double>(msgs.size());
          return JsonOk(out);
        }
        return JsonErr(404, "Not Found", "Conversation not found");
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }

    // GET /api/conversations/<id>
    if (sub_route.empty() && req.method == "GET") {
      const auto conv = db.GetConversation(user->id, conv_id);
      if (!conv.has_value()) {
        return JsonErr(404, "Not Found", "Conversation not found");
      }
      json::Value obj = json::Value::object();
      obj["id"] = conv->id;
      obj["title"] = conv->title;
      obj["model"] = conv->model;
      obj["system_prompt"] = conv->system_prompt;
      obj["working_directory"] = conv->working_directory;
      obj["created_at"] = conv->created_at;
      obj["updated_at"] = conv->updated_at;

      json::Value msgs_arr = json::Value::array();
      for (const auto& m : conv->messages) {
        json::Value mobj = json::Value::object();
        mobj["id"] = m.id;
        mobj["role"] = m.role;
        mobj["content"] = m.content;
        mobj["reasoning_content"] = m.reasoning_content;
        mobj["created_at"] = m.created_at;
        msgs_arr.push_back(std::move(mobj));
      }
      obj["messages"] = std::move(msgs_arr);
      return JsonOk(obj);
    }

    // PATCH /api/conversations/<id>
    if (sub_route.empty() && req.method == "PATCH") {
      try {
        const auto body = json::parse(req.body);
        bool updated = false;
        if (const auto* w = body.find("working_directory"); w && w->is_string()) {
          updated = db.UpdateConversationWorkingDir(user->id, conv_id, w->str()) || updated;
        }
        if (const auto* t = body.find("title"); t && t->is_string() && !t->str().empty()) {
          updated = db.UpdateConversationTitle(user->id, conv_id, t->str()) || updated;
        }
        if (const auto* sp = body.find("system_prompt"); sp && sp->is_string()) {
          updated = db.UpdateConversationSystemPrompt(user->id, conv_id, sp->str()) || updated;
        }
        if (updated) {
          const auto updated_conv = db.GetConversation(user->id, conv_id);
          json::Value out = json::Value::object();
          out["status"] = "ok";
          if (updated_conv) {
            out["working_directory"] = updated_conv->working_directory;
            out["title"] = updated_conv->title;
            out["system_prompt"] = updated_conv->system_prompt;
          }
          return JsonOk(out);
        }
        return JsonErr(404, "Not Found", "Conversation not found or nothing updated");
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }

    // DELETE /api/conversations/<id>
    if (sub_route.empty() && req.method == "DELETE") {
      if (db.DeleteConversation(user->id, conv_id)) {
        json::Value out = json::Value::object();
        out["status"] = "ok";
        return JsonOk(out);
      }
      return JsonErr(404, "Not Found", "Conversation not found");
    }
  }

  // --- Settings APIs ---
  if (req.path == "/api/settings") {
    if (req.method == "GET") {
      const auto st = db.GetSettings(user->id);
      json::Value obj = json::Value::object();
      obj["temperature"] = st.temperature;
      obj["top_p"] = st.top_p;
      obj["max_tokens"] = st.max_tokens;
      obj["system_prompt"] = st.system_prompt;
      obj["working_directory"] = st.working_directory;
      return JsonOk(obj);
    }
    if (req.method == "POST" || req.method == "PUT") {
      try {
        const auto body = json::parse(req.body);
        UserSettingsRecord st = db.GetSettings(user->id);
        st.user_id = user->id;
        if (const auto* t = body.find("temperature"); t && t->is_number())
          st.temperature = t->as_double();
        if (const auto* p = body.find("top_p"); p && p->is_number())
          st.top_p = p->as_double();
        if (const auto* m = body.find("max_tokens"); m && m->is_number())
          st.max_tokens = static_cast<int>(m->as_double());
        if (const auto* s = body.find("system_prompt"); s && s->is_string())
          st.system_prompt = s->str();
        if (const auto* w = body.find("working_directory"); w && w->is_string())
          st.working_directory = w->str();
        db.SaveSettings(st);

        json::Value out = json::Value::object();
        out["status"] = "ok";
        return JsonOk(out);
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }
  }

  // --- Model Live Monitoring API ---
  if (req.path == "/api/monitor" && req.method == "GET") {
    json::Value res = json::Value::object();
    res["messages_processing"] = static_cast<std::int64_t>(
        detail::RequestsProcessing().load(std::memory_order_relaxed));
    res["messages_deferred"] = static_cast<std::int64_t>(
        detail::RequestsDeferred().load(std::memory_order_relaxed));
    res["last_prompt_speed"] =
        detail::LastPromptSpeed().load(std::memory_order_relaxed);
    res["last_gen_speed"] =
        detail::LastGenSpeed().load(std::memory_order_relaxed);
    res["last_prompt_tokens"] = static_cast<std::uint64_t>(
        detail::LastPromptTokens().load(std::memory_order_relaxed));
    res["last_gen_tokens"] = static_cast<std::uint64_t>(
        detail::LastGenTokens().load(std::memory_order_relaxed));
    res["total_prompt_tokens"] = static_cast<std::uint64_t>(
        detail::TotalPromptTokens().load(std::memory_order_relaxed));
    res["total_gen_tokens"] = static_cast<std::uint64_t>(
        detail::TotalGenTokens().load(std::memory_order_relaxed));
    res["total_cached_prompt_tokens"] = static_cast<std::uint64_t>(
        detail::TotalCachedPromptTokens().load(std::memory_order_relaxed));
    res["total_draft_tokens"] = static_cast<std::uint64_t>(
        detail::TotalDraftTokens().load(std::memory_order_relaxed));
    res["total_draft_accepted_tokens"] = static_cast<std::uint64_t>(
        detail::TotalDraftAcceptedTokens().load(std::memory_order_relaxed));
    res["total_draft_rounds"] = static_cast<std::uint64_t>(
        detail::TotalDraftRounds().load(std::memory_order_relaxed));
    res["memory_status"] = Logger::MemoryStatus();
    return JsonResponse(200, "OK", res.dump());
  }

  // --- Tools & MCP APIs ---
  if (req.path == "/api/tools" && req.method == "GET") {
    json::Value out = json::Value::object();

    // 1. Inbuilt tools + config
    const std::string conf_str = db.GetInbuiltToolsConfig(user->id);
    json::Value user_conf = json::Value::object();
    try { user_conf = json::parse(conf_str); } catch (...) {}

    bool has_enabled_list = false;
    std::vector<std::string> enabled_list;
    if (const auto* arr = user_conf.find("enabled"); arr && arr->is_array()) {
      has_enabled_list = true;
      for (const auto& item : arr->items()) {
        enabled_list.push_back(item.str());
      }
    }

    bool master_enabled = true;
    if (const auto* me = user_conf.find("master_enabled"); me && me->is_bool()) {
      master_enabled = me->as_bool();
    }

    json::Value inbuilt_arr = json::Value::array();
    for (const auto& t : ToolEngine::GetInbuiltTools()) {
      json::Value tobj = json::Value::object();
      tobj["name"] = t.name;
      tobj["description"] = t.description;
      tobj["parameters"] = t.parameters;
      tobj["dangerous"] = t.dangerous;
      bool enabled = t.enabled_by_default;
      if (has_enabled_list) {
        enabled = std::find(enabled_list.begin(), enabled_list.end(), t.name) != enabled_list.end();
      } else if (const auto* p = user_conf.find(t.name); p && p->is_bool()) {
        enabled = p->as_bool();
      }
      tobj["enabled"] = enabled;
      inbuilt_arr.push_back(std::move(tobj));
    }
    out["inbuilt"] = std::move(inbuilt_arr);
    out["master_enabled"] = master_enabled;
    std::error_code ec;
    out["server_current_directory"] = std::filesystem::current_path(ec).string();
    out["working_directory"] = db.GetSettings(user->id).working_directory;

    // 2. MCP servers
    const auto servers = db.ListMcpServers(user->id);
    json::Value srv_arr = json::Value::array();
    for (const auto& s : servers) {
      json::Value sobj = json::Value::object();
      sobj["id"] = s.id;
      sobj["name"] = s.name;
      sobj["url"] = s.url;
      sobj["transport"] = s.transport;
      sobj["headers_json"] = s.headers_json;
      sobj["tools_cache"] = s.tools_cache_json;
      sobj["enabled"] = s.enabled;
      sobj["created_at"] = s.created_at;
      srv_arr.push_back(std::move(sobj));
    }
    out["mcp_servers"] = std::move(srv_arr);

    // 3. Custom tools
    const auto custom = db.ListCustomTools(user->id);
    json::Value cst_arr = json::Value::array();
    for (const auto& c : custom) {
      json::Value cobj = json::Value::object();
      cobj["id"] = c.id;
      cobj["name"] = c.name;
      cobj["description"] = c.description;
      cobj["schema_json"] = c.schema_json;
      cobj["handler_url"] = c.handler_url;
      cobj["enabled"] = c.enabled;
      cobj["created_at"] = c.created_at;
      cst_arr.push_back(std::move(cobj));
    }
    out["custom_tools"] = std::move(cst_arr);

    return JsonOk(out);
  }

  if (req.path == "/api/tools/inbuilt" && (req.method == "POST" || req.method == "PUT")) {
    try {
      const auto body = json::parse(req.body);
      db.SaveInbuiltToolsConfig(user->id, body.dump());
      json::Value out = json::Value::object();
      out["status"] = "ok";
      return JsonOk(out);
    } catch (...) {
      return JsonErr(400, "Bad Request", "Invalid JSON payload");
    }
  }

  if (req.path == "/api/tools/execute" && req.method == "POST") {
    try {
      const auto body = json::parse(req.body);
      const std::string name = body.contains("name") ? body.member_str("name") : body.member_str("tool_name");
      const auto* args_ptr = body.find("arguments");
      json::Value args = (args_ptr && args_ptr->is_object()) ? *args_ptr : json::Value::object();
      std::string working_dir = body.member_str("working_directory");
      const std::string conv_id = body.member_str("conversation_id");
      if (working_dir.empty() && !conv_id.empty()) {
        const auto conv = db.GetConversation(user->id, conv_id);
        if (conv) {
          working_dir = conv->working_directory;
        }
      }
      if (working_dir.empty()) {
        const auto st = db.GetSettings(user->id);
        working_dir = st.working_directory;
      }
      if (working_dir.empty() && !conv_id.empty()) {
        working_dir = GetConversationDefaultWorkingDir(conv_id, "default").string();
      }

      // Check if inbuilt tool
      if (ToolEngine::FindInbuiltTool(name) != nullptr) {
        const auto res = ToolEngine::Execute(name, args, working_dir);
        json::Value out = json::Value::object();
        out["success"] = res.success;
        out["output"] = res.output;
        out["result"] = res.output.empty() ? res.error : res.output;
        out["error"] = res.error;
        return JsonOk(out);
      }

      // Check if custom tool
      const auto custom = db.ListCustomTools(user->id);
      for (const auto& c : custom) {
        if (c.name == name) {
          if (!c.handler_url.empty()) {
            const auto res = ToolEngine::ExecuteMcpProxy(c.handler_url, "{}", name, args);
            json::Value out = json::Value::object();
            out["success"] = res.success;
            out["output"] = res.output;
            out["error"] = res.error;
            return JsonOk(out);
          }
          break;
        }
      }

      // Check if MCP server provides it
      const auto servers = db.ListMcpServers(user->id);
      for (const auto& s : servers) {
        if (!s.enabled) continue;
        try {
          const auto tools = json::parse(s.tools_cache_json);
          if (tools.is_array()) {
            for (const auto& item : tools.items()) {
              if (item.member_str("name") == name) {
                const auto res = ToolEngine::ExecuteMcpProxy(s.url, s.headers_json, name, args);
                json::Value out = json::Value::object();
                out["success"] = res.success;
                out["output"] = res.output;
                out["error"] = res.error;
                return JsonOk(out);
              }
            }
          }
        } catch (...) {}
      }

      return JsonErr(404, "Not Found", "Tool not found: " + name);
    } catch (...) {
      return JsonErr(400, "Bad Request", "Invalid JSON payload");
    }
  }

  // --- MCP Servers APIs ---
  if (req.path == "/api/mcp/servers") {
    if (req.method == "GET") {
      const auto servers = db.ListMcpServers(user->id);
      json::Value arr = json::Value::array();
      for (const auto& s : servers) {
        json::Value obj = json::Value::object();
        obj["id"] = s.id;
        obj["name"] = s.name;
        obj["url"] = s.url;
        obj["transport"] = s.transport;
        obj["headers_json"] = s.headers_json;
        obj["tools_cache"] = s.tools_cache_json;
        obj["enabled"] = s.enabled;
        obj["created_at"] = s.created_at;
        arr.push_back(std::move(obj));
      }
      return JsonOk(arr);
    }
    if (req.method == "POST") {
      try {
        const auto body = json::parse(req.body);
        const std::string name = body.member_str("name");
        const std::string url = body.member_str("url");
        const std::string transport = body.contains("transport") ? body.member_str("transport") : "sse";
        const auto* h = body.find("headers");
        const std::string headers = h ? h->dump() : "{}";
        const auto srv = db.AddMcpServer(user->id, name, url, transport, headers);
        json::Value obj = json::Value::object();
        obj["id"] = srv.id;
        obj["name"] = srv.name;
        obj["url"] = srv.url;
        obj["transport"] = srv.transport;
        obj["headers_json"] = srv.headers_json;
        obj["tools_cache"] = srv.tools_cache_json;
        obj["enabled"] = srv.enabled;
        obj["created_at"] = srv.created_at;
        return JsonOk(obj);
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }
  }

  if (req.path.starts_with("/api/mcp/servers/")) {
    const std::string rest = req.path.substr(std::string("/api/mcp/servers/").size());
    const auto slash = rest.find('/');
    const std::string server_id = slash == std::string::npos ? rest : rest.substr(0, slash);
    const std::string sub_route = slash == std::string::npos ? "" : rest.substr(slash);

    if (sub_route == "/sync" && req.method == "POST") {
      const auto servers = db.ListMcpServers(user->id);
      for (auto s : servers) {
        if (s.id == server_id) {
          const auto res = ToolEngine::SyncMcpTools(s.url, s.headers_json);
          if (!res.success) {
            return JsonErr(502, "Bad Gateway", res.error);
          }
          s.tools_cache_json = res.output;
          db.UpdateMcpServer(user->id, s);
          json::Value out = json::Value::object();
          out["tools"] = res.output;
          return JsonOk(out);
        }
      }
      return JsonErr(404, "Not Found", "MCP server not found");
    }

    if (sub_route.empty() && req.method == "PATCH") {
      try {
        const auto body = json::parse(req.body);
        const auto servers = db.ListMcpServers(user->id);
        for (auto s : servers) {
          if (s.id == server_id) {
            if (body.contains("name")) s.name = body.member_str("name");
            if (body.contains("url")) s.url = body.member_str("url");
            if (body.contains("transport")) s.transport = body.member_str("transport");
            if (body.contains("headers_json")) s.headers_json = body.member_str("headers_json");
            if (const auto* en = body.find("enabled"); en && en->is_bool()) s.enabled = en->as_bool();
            db.UpdateMcpServer(user->id, s);
            json::Value out = json::Value::object();
            out["status"] = "ok";
            return JsonOk(out);
          }
        }
        return JsonErr(404, "Not Found", "MCP server not found");
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }

    if (sub_route.empty() && req.method == "DELETE") {
      if (db.DeleteMcpServer(user->id, server_id)) {
        json::Value out = json::Value::object();
        out["status"] = "ok";
        return JsonOk(out);
      }
      return JsonErr(404, "Not Found", "MCP server not found");
    }
  }

  // --- Custom Tools APIs ---
  if (req.path == "/api/custom-tools") {
    if (req.method == "GET") {
      const auto tools = db.ListCustomTools(user->id);
      json::Value arr = json::Value::array();
      for (const auto& t : tools) {
        json::Value obj = json::Value::object();
        obj["id"] = t.id;
        obj["name"] = t.name;
        obj["description"] = t.description;
        obj["schema_json"] = t.schema_json;
        obj["handler_url"] = t.handler_url;
        obj["enabled"] = t.enabled;
        obj["created_at"] = t.created_at;
        arr.push_back(std::move(obj));
      }
      return JsonOk(arr);
    }
    if (req.method == "POST") {
      try {
        const auto body = json::parse(req.body);
        const std::string name = body.member_str("name");
        const std::string description = body.member_str("description");
        const auto* params = body.find("parameters");
        const std::string schema = params ? params->dump() : body.member_str("schema_json", "{}");
        const std::string handler_url = body.member_str("handler_url", "");
        const auto t = db.AddCustomTool(user->id, name, description, schema, handler_url);
        json::Value obj = json::Value::object();
        obj["id"] = t.id;
        obj["name"] = t.name;
        obj["description"] = t.description;
        obj["schema_json"] = t.schema_json;
        obj["handler_url"] = t.handler_url;
        obj["enabled"] = t.enabled;
        obj["created_at"] = t.created_at;
        return JsonOk(obj);
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }
  }

  if (req.path.starts_with("/api/custom-tools/")) {
    const std::string tool_id = req.path.substr(std::string("/api/custom-tools/").size());
    if (req.method == "DELETE") {
      if (db.DeleteCustomTool(user->id, tool_id)) {
        json::Value out = json::Value::object();
        out["status"] = "ok";
        return JsonOk(out);
      }
      return JsonErr(404, "Not Found", "Custom tool not found");
    }
  }

  return JsonErr(404, "Not Found", "API endpoint not found");
}

}  // namespace gufo::server
