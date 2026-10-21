#ifndef GUFO_SERVER_CHAT_PORTAL_SERVICE_HPP_
#define GUFO_SERVER_CHAT_PORTAL_SERVICE_HPP_

#include "src/cli/serve/chat_portal_db.hpp"
#include "src/cli/serve/http_server.hpp"

namespace gufo::server {

class ChatPortalService {
public:
  static bool IsPortalRoute(const HttpRequest& req);
  static HttpResponse Handle(
      const HttpRequest& req, ChatDatabase& db,
      std::shared_ptr<TextGenerationBackend> backend = nullptr,
      std::optional<bool> tools_enabled = std::nullopt);
  static std::string ExtractBearerToken(const HttpRequest& req);
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CHAT_PORTAL_SERVICE_HPP_
