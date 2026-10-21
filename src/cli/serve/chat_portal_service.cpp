#include "src/cli/serve/chat_portal_service.hpp"

#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>

#include "src/cli/serve/chat_portal_html.hpp"
#include "src/cli/serve/chat_portal_tools.hpp"
#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/openai_chat.hpp"
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

std::string UrlDecodeSimple(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      int h1 = hex(s[i + 1]);
      int h2 = hex(s[i + 2]);
      if (h1 >= 0 && h2 >= 0) {
        out.push_back(static_cast<char>((h1 << 4) | h2));
        i += 2;
        continue;
      }
    } else if (s[i] == '+') {
      out.push_back(' ');
      continue;
    }
    out.push_back(s[i]);
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

struct ActiveConversationJob : public std::enable_shared_from_this<ActiveConversationJob> {
  std::string conv_id;
  std::string user_id;
  bool is_resume{false};
  std::atomic<bool> cancelled{false};
  std::atomic<bool> completed{false};

  std::mutex mutex;
  std::vector<std::string> event_history;
  std::string accumulated_content;
  std::string accumulated_thought;

  struct Subscriber {
    std::uint64_t id{0};
    std::shared_ptr<std::mutex> sub_mutex;
    std::shared_ptr<std::condition_variable> sub_cv;
    std::shared_ptr<std::vector<std::string>> queue;
    std::shared_ptr<std::atomic<bool>> disconnected;
  };
  std::vector<Subscriber> subscribers;
  std::uint64_t next_sub_id{1};

  std::jthread worker_thread;

  void Cancel() {
    cancelled.store(true);
  }

  void BroadcastEvent(std::string_view event_str) {
    std::lock_guard<std::mutex> lock(mutex);
    event_history.emplace_back(event_str);
    ParseAndAccumulateDelta(event_str);

    for (auto it = subscribers.begin(); it != subscribers.end(); ) {
      if (it->disconnected->load()) {
        it = subscribers.erase(it);
      } else {
        {
          std::lock_guard<std::mutex> sl(*it->sub_mutex);
          it->queue->emplace_back(event_str);
        }
        it->sub_cv->notify_one();
        ++it;
      }
    }
  }

  void StreamToClient(const HttpResponse::BodyWriter& writer) {
    auto sub_mutex = std::make_shared<std::mutex>();
    auto sub_cv = std::make_shared<std::condition_variable>();
    auto queue = std::make_shared<std::vector<std::string>>();
    auto disconnected = std::make_shared<std::atomic<bool>>(false);
    std::uint64_t sub_id = 0;
    std::vector<std::string> backlog;
    bool already_completed = false;

    {
      std::lock_guard<std::mutex> lock(mutex);
      backlog = event_history;
      already_completed = completed.load();
      if (!already_completed) {
        sub_id = next_sub_id++;
        subscribers.push_back({sub_id, sub_mutex, sub_cv, queue, disconnected});
      }
    }

    // 1. Send all past backlog events
    for (const auto& ev : backlog) {
      if (!writer(ev)) {
        disconnected->store(true);
        RemoveSubscriber(sub_id);
        return;
      }
    }

    if (already_completed) return;

    // 2. Loop and stream new incoming events
    while (!disconnected->load()) {
      std::vector<std::string> pending;
      {
        std::unique_lock<std::mutex> sl(*sub_mutex);
        sub_cv->wait_for(sl, std::chrono::milliseconds(250), [&] {
          return !queue->empty() || completed.load() || disconnected->load();
        });
        if (disconnected->load()) break;
        pending.swap(*queue);
      }

      for (const auto& ev : pending) {
        if (!writer(ev)) {
          disconnected->store(true);
          RemoveSubscriber(sub_id);
          return;
        }
      }

      if (completed.load() && pending.empty()) {
        break;
      }
    }

    RemoveSubscriber(sub_id);
  }

  void RemoveSubscriber(std::uint64_t id) {
    if (id == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = subscribers.begin(); it != subscribers.end(); ++it) {
      if (it->id == id) {
        subscribers.erase(it);
        break;
      }
    }
  }

  void OnFinished(ChatDatabase& db) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      completed.store(true);
      const std::string done_ev = "data: [DONE]\n\n";
      bool has_done = !event_history.empty() && event_history.back() == done_ev;
      if (!has_done) {
        event_history.push_back(done_ev);
        for (const auto& sub : subscribers) {
          {
            std::lock_guard<std::mutex> sl(*sub.sub_mutex);
            sub.queue->push_back(done_ev);
          }
          sub.sub_cv->notify_one();
        }
      }
    }

    // Save to database
    if (!accumulated_content.empty() || !accumulated_thought.empty()) {
      if (is_resume) {
        db.UpdateLastAssistantMessage(user_id, conv_id, accumulated_content, accumulated_thought);
      } else {
        ChatMessageRecord msg;
        msg.role = "assistant";
        msg.content = accumulated_content;
        msg.reasoning_content = accumulated_thought;
        db.AddMessage(user_id, conv_id, msg);
      }
    }
  }

  void ParseAndAccumulateDelta(std::string_view event_str) {
    std::string_view s = event_str;
    while (!s.empty() && (s.front() == ' ' || s.front() == '\r' || s.front() == '\n')) s.remove_prefix(1);
    if (!s.starts_with("data:")) return;
    s.remove_prefix(5);
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r' || s.back() == '\n')) s.remove_suffix(1);
    if (s == "[DONE]" || s.empty()) return;

    try {
      const auto chunk = json::parse(s);
      if (const auto* choices = chunk.find("choices"); choices && choices->is_array() && !choices->empty()) {
        if (const auto* delta = choices->items().front().find("delta"); delta && delta->is_object()) {
          if (const auto* r = delta->find("reasoning_content"); r && r->is_string()) {
            accumulated_thought.append(r->str());
          }
          if (const auto* t = delta->find("thought"); t && t->is_string()) {
            accumulated_thought.append(t->str());
          }
          if (const auto* c = delta->find("content"); c && c->is_string()) {
            accumulated_content.append(c->str());
          }
        }
      }
    } catch (...) {}
  }
};

class ConversationGenerationManager {
 public:
  static ConversationGenerationManager& Instance() {
    static ConversationGenerationManager inst;
    return inst;
  }

  std::shared_ptr<ActiveConversationJob> GetJob(const std::string& conv_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(conv_id);
    if (it != jobs_.end()) {
      return it->second;
    }
    return nullptr;
  }

  bool IsGenerating(const std::string& conv_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(conv_id);
    if (it != jobs_.end()) {
      return !it->second->completed.load();
    }
    return false;
  }

  std::shared_ptr<ActiveConversationJob> CreateJob(const std::string& conv_id,
                                                  const std::string& user_id,
                                                  bool is_resume) {
    std::lock_guard<std::mutex> lock(mutex_);
    CleanupFinishedJobsLocked();
    auto job = std::make_shared<ActiveConversationJob>();
    job->conv_id = conv_id;
    job->user_id = user_id;
    job->is_resume = is_resume;
    jobs_[conv_id] = job;
    return job;
  }

  void RemoveJob(const std::string& conv_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.erase(conv_id);
  }

 private:
  ConversationGenerationManager() = default;

  void CleanupFinishedJobsLocked() {
    for (auto it = jobs_.begin(); it != jobs_.end(); ) {
      if (it->second->completed.load()) {
        it = jobs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<ActiveConversationJob>> jobs_;
};

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
                                       ChatDatabase& db,
                                       std::shared_ptr<TextGenerationBackend> backend,
                                       std::optional<bool> tools_enabled) {
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
      obj["is_generating"] = ConversationGenerationManager::Instance().IsGenerating(c.id);
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

    // POST /api/conversations/<id>/generate
    if (sub_route == "/generate" && req.method == "POST") {
      if (!backend || !backend->ready()) {
        return JsonErr(503, "Service Unavailable", "Inference backend is not ready");
      }

      auto& manager = ConversationGenerationManager::Instance();
      auto existing_job = manager.GetJob(conv_id);
      if (existing_job && !existing_job->completed.load()) {
        HttpResponse resp;
        resp.status = 200;
        resp.reason = "OK";
        resp.headers.emplace_back("Content-Type", "text/event-stream");
        resp.headers.emplace_back("Cache-Control", "no-cache");
        resp.headers.emplace_back("X-Accel-Buffering", "no");
        resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
        resp.streaming_body = [existing_job](const HttpResponse::BodyWriter& writer) {
          existing_job->StreamToClient(writer);
        };
        return resp;
      }

      bool is_resume = false;
      try {
        const auto b = json::parse(req.body);
        if (const auto* r = b.find("is_resume"); r && r->is_bool()) {
          is_resume = r->as_bool();
        }
      } catch (...) {}

      auto job = manager.CreateJob(conv_id, user->id, is_resume);
      if (is_resume) {
        const auto conv = db.GetConversation(user->id, conv_id);
        if (conv) {
          for (auto it = conv->messages.rbegin(); it != conv->messages.rend(); ++it) {
            if (it->role == "assistant") {
              job->accumulated_content = it->content;
              job->accumulated_thought = it->reasoning_content;
              break;
            }
          }
        }
      }

      HttpRequest chat_req = req;
      chat_req.path = "/v1/chat/completions";
      chat_req.is_cancelled = [job] { return job->cancelled.load(); };

      HttpResponse chat_resp = HandleOpenAiChat(chat_req, *backend);
      if (chat_resp.status != 200 || !chat_resp.streaming_body) {
        manager.RemoveJob(conv_id);
        return chat_resp;
      }

      auto body_fn = std::move(chat_resp.streaming_body);
      job->worker_thread = std::jthread([job, body_fn = std::move(body_fn), &db]() mutable {
        try {
          body_fn([job](std::string_view chunk) -> bool {
            if (job->cancelled.load()) return false;
            job->BroadcastEvent(chunk);
            return true;
          });
        } catch (...) {}
        job->OnFinished(db);
      });

      HttpResponse resp;
      resp.status = 200;
      resp.reason = "OK";
      resp.headers.emplace_back("Content-Type", "text/event-stream");
      resp.headers.emplace_back("Cache-Control", "no-cache");
      resp.headers.emplace_back("X-Accel-Buffering", "no");
      resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
      resp.streaming_body = [job](const HttpResponse::BodyWriter& writer) {
        job->StreamToClient(writer);
      };
      return resp;
    }

    // GET /api/conversations/<id>/stream
    if (sub_route == "/stream" && req.method == "GET") {
      auto job = ConversationGenerationManager::Instance().GetJob(conv_id);
      if (!job) {
        HttpResponse resp;
        resp.status = 200;
        resp.reason = "OK";
        resp.headers.emplace_back("Content-Type", "text/event-stream");
        resp.headers.emplace_back("Cache-Control", "no-cache");
        resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
        resp.body = "data: [DONE]\n\n";
        return resp;
      }

      HttpResponse resp;
      resp.status = 200;
      resp.reason = "OK";
      resp.headers.emplace_back("Content-Type", "text/event-stream");
      resp.headers.emplace_back("Cache-Control", "no-cache");
      resp.headers.emplace_back("X-Accel-Buffering", "no");
      resp.headers.emplace_back("Access-Control-Allow-Origin", "*");
      resp.streaming_body = [job](const HttpResponse::BodyWriter& writer) {
        job->StreamToClient(writer);
      };
      return resp;
    }

    // POST /api/conversations/<id>/stop
    if (sub_route == "/stop" && req.method == "POST") {
      auto job = ConversationGenerationManager::Instance().GetJob(conv_id);
      if (job) {
        job->Cancel();
      }
      json::Value out = json::Value::object();
      out["status"] = "ok";
      return JsonOk(out);
    }

    // GET /api/conversations/<id>/status
    if (sub_route == "/status" && req.method == "GET") {
      const bool is_gen = ConversationGenerationManager::Instance().IsGenerating(conv_id);
      json::Value out = json::Value::object();
      out["is_generating"] = is_gen;
      if (is_gen) {
        auto job = ConversationGenerationManager::Instance().GetJob(conv_id);
        if (job) out["is_resume"] = job->is_resume;
      }
      return JsonOk(out);
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
      const bool is_gen = ConversationGenerationManager::Instance().IsGenerating(conv_id);
      obj["is_generating"] = is_gen;
      if (is_gen) {
        auto job = ConversationGenerationManager::Instance().GetJob(conv_id);
        if (job) obj["is_resume"] = job->is_resume;
      }

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
    if (tools_enabled.has_value()) {
      master_enabled = *tools_enabled;
      out["server_tools_override"] = true;
    }

    json::Value inbuilt_arr = json::Value::array();
    for (const auto& t : ToolEngine::GetInbuiltTools()) {
      if (t.name == "ask_user") {
        continue;
      }
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
      if (tools_enabled.has_value() && !*tools_enabled && name != "ask_user") {
        return JsonErr(403, "Forbidden", "Tool execution is disabled by server --tools flag");
      }
      const auto* args_ptr = body.find("arguments");
      json::Value args = (args_ptr && args_ptr->is_object()) ? *args_ptr : json::Value::object();
      std::string working_dir = body.member_str("working_directory");
      const std::string conv_id = body.member_str("conversation_id");
      const auto user_root = GetUserRootWorkingDir(user->id, user->username);
      if (working_dir.empty() && !conv_id.empty()) {
        const auto conv = db.GetConversation(user->id, conv_id);
        if (conv && !conv->working_directory.empty()) {
          working_dir = conv->working_directory;
        } else {
          working_dir = GetConversationDefaultFolderName(conv_id, conv ? conv->title : "Chat");
        }
      }
      if (working_dir.empty()) {
        const auto st = db.GetSettings(user->id);
        working_dir = st.working_directory;
      }

      std::filesystem::path full_wd;
      if (std::filesystem::path(working_dir).is_absolute()) {
        full_wd = std::filesystem::path(working_dir);
      } else if (!working_dir.empty()) {
        full_wd = user_root / working_dir;
      } else {
        full_wd = user_root;
      }
      std::error_code ec;
      std::filesystem::create_directories(full_wd, ec);
      if (!IsPathInsideDirectory(full_wd, user_root)) {
        working_dir = "";
      }

      // Check if inbuilt tool
      if (ToolEngine::FindInbuiltTool(name) != nullptr) {
        const auto res = ToolEngine::Execute(name, args, working_dir, user_root.string());
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

  // -------------------------------------------------------------
  // User Files API (Sandboxed to UserData/{user_id}_{username}/)
  // -------------------------------------------------------------
  if (req.path == "/api/user-files") {
    const auto user_root = GetUserRootWorkingDir(user->id, user->username);

    if (req.method == "GET") {
      std::string subpath;
      auto q = req.query.find("subpath=");
      if (q != std::string::npos) {
        subpath = req.query.substr(q + 8);
      } else {
        q = req.query.find("path=");
        if (q != std::string::npos) {
          subpath = req.query.substr(q + 5);
        }
      }
      const auto amp = subpath.find('&');
      if (amp != std::string::npos) subpath = subpath.substr(0, amp);
      subpath = UrlDecodeSimple(subpath);

      auto target = user_root;
      if (!subpath.empty()) {
        target = user_root / std::filesystem::path(subpath);
      }
      std::error_code ec;
      target = std::filesystem::weakly_canonical(target, ec);
      if (ec || !IsPathInsideDirectory(target, user_root)) {
        return JsonErr(403, "Forbidden", "Access outside user directory is prohibited");
      }

      json::Value arr = json::Value::array();
      if (std::filesystem::exists(target, ec) && std::filesystem::is_directory(target, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(target, ec)) {
          json::Value item = json::Value::object();
          item["name"] = entry.path().filename().string();
          item["is_dir"] = entry.is_directory(ec);
          item["size"] = entry.is_regular_file(ec) ? static_cast<double>(entry.file_size(ec)) : 0.0;
          auto rel = std::filesystem::relative(entry.path(), user_root, ec);
          item["rel_path"] = rel.generic_string();
          arr.push_back(std::move(item));
        }
      }

      // Collect all folder relative paths under user_root for working directory selection
      json::Value folders_list = json::Value::array();
      if (std::filesystem::exists(user_root, ec)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(user_root, std::filesystem::directory_options::skip_permission_denied, ec)) {
          if (entry.is_directory(ec)) {
            auto rel = std::filesystem::relative(entry.path(), user_root, ec);
            if (!rel.empty()) {
              folders_list.push_back(rel.generic_string());
            }
          }
        }
      }

      json::Value out = json::Value::object();
      auto cur_rel = std::filesystem::relative(target, user_root, ec);
      const std::string cur_str = (cur_rel.empty() || cur_rel == ".") ? "" : cur_rel.generic_string();
      const std::string username_display = user->username.empty() ? "Guest" : user->username;
      out["current_path"] = cur_str;
      out["user_root_name"] = username_display;
      out["logical_path"] = "UserData/" + username_display + (cur_str.empty() ? "" : ("/" + cur_str));
      out["entries"] = std::move(arr);
      out["folders_list"] = std::move(folders_list);
      return JsonOk(out);
    }

    if (req.method == "DELETE") {
      std::string rel_path;
      auto q = req.query.find("path=");
      if (q != std::string::npos) {
        rel_path = req.query.substr(q + 5);
        auto amp = rel_path.find('&');
        if (amp != std::string::npos) rel_path = rel_path.substr(0, amp);
        rel_path = UrlDecodeSimple(rel_path);
      }
      if (rel_path.empty() && !req.body.empty()) {
        try {
          const auto body = json::parse(req.body);
          rel_path = body.member_str("path");
        } catch (...) {}
      }
      if (rel_path.empty()) return JsonErr(400, "Bad Request", "Missing 'path'");
      auto target = user_root / std::filesystem::path(rel_path);
      std::error_code ec;
      target = std::filesystem::weakly_canonical(target, ec);
      if (ec || !IsPathInsideDirectory(target, user_root) || target == user_root) {
        return JsonErr(403, "Forbidden", "Access outside user directory is prohibited");
      }
      std::filesystem::remove_all(target, ec);
      json::Value out = json::Value::object();
      out["status"] = "ok";
      return JsonOk(out);
    }
  }

  // -------------------------------------------------------------
  // User Folders API (Create directories under UserData/<user>/)
  // -------------------------------------------------------------
  if (req.path == "/api/user-folders" && req.method == "POST") {
    const auto user_root = GetUserRootWorkingDir(user->id, user->username);
    try {
      const auto body = json::parse(req.body);
      const std::string name = body.member_str("name");
      const std::string parent = body.member_str("parent");
      if (name.empty()) return JsonErr(400, "Bad Request", "Missing 'name'");

      auto target = user_root;
      if (!parent.empty()) target = target / std::filesystem::path(parent);
      target = target / std::filesystem::path(name);

      std::error_code ec;
      target = std::filesystem::weakly_canonical(target, ec);
      if (ec) target = target.lexically_normal();
      if (!IsPathInsideDirectory(target, user_root)) {
        return JsonErr(403, "Forbidden", "Destination is outside user directory");
      }
      std::filesystem::create_directories(target, ec);
      if (ec) return JsonErr(500, "Internal Error", "Failed to create directory");

      json::Value out = json::Value::object();
      auto rel = std::filesystem::relative(target, user_root, ec);
      out["status"] = "ok";
      out["path"] = rel.generic_string();
      out["name"] = target.filename().string();
      return JsonOk(out);
    } catch (...) {
      return JsonErr(400, "Bad Request", "Invalid JSON payload");
    }
  }

  if (req.path == "/api/user-files/content") {
    const auto user_root = GetUserRootWorkingDir(user->id, user->username);
    if (req.method == "GET") {
      std::string subpath;
      auto q = req.query.find("path=");
      if (q != std::string::npos) {
        subpath = req.query.substr(q + 5);
        const auto amp = subpath.find('&');
        if (amp != std::string::npos) subpath = subpath.substr(0, amp);
        subpath = UrlDecodeSimple(subpath);
      }
      if (subpath.empty()) return JsonErr(400, "Bad Request", "Missing 'path'");
      auto target = user_root / std::filesystem::path(subpath);
      std::error_code ec;
      target = std::filesystem::weakly_canonical(target, ec);
      if (ec || !IsPathInsideDirectory(target, user_root) || !std::filesystem::is_regular_file(target, ec)) {
        return JsonErr(403, "Forbidden", "File inaccessible or outside user directory");
      }
      std::ifstream in(target, std::ios::binary);
      if (!in.is_open()) return JsonErr(500, "Internal Error", "Failed to open file");
      std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      json::Value out = json::Value::object();
      out["name"] = target.filename().string();
      out["content"] = content;
      out["size"] = static_cast<double>(content.size());
      return JsonOk(out);
    }
    if (req.method == "POST") {
      try {
        const auto body = json::parse(req.body);
        const std::string rel_path = body.member_str("path");
        const std::string content = body.member_str("content");
        if (rel_path.empty()) return JsonErr(400, "Bad Request", "Missing 'path'");
        auto target = user_root / std::filesystem::path(rel_path);
        std::error_code ec;
        target = std::filesystem::weakly_canonical(target, ec);
        if (ec) target = (user_root / std::filesystem::path(rel_path)).lexically_normal();
        if (!IsPathInsideDirectory(target, user_root)) {
          return JsonErr(403, "Forbidden", "Destination is outside user directory");
        }
        if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path(), ec);
        std::ofstream out_f(target, std::ios::binary | std::ios::trunc);
        if (!out_f.is_open()) return JsonErr(500, "Internal Error", "Cannot write file");
        out_f << content;
        json::Value out = json::Value::object();
        out["status"] = "ok";
        return JsonOk(out);
      } catch (...) {
        return JsonErr(400, "Bad Request", "Invalid JSON payload");
      }
    }
  }

  return JsonErr(404, "Not Found", "API endpoint not found");
}

}  // namespace gufo::server
