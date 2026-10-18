#ifndef GUFO_SERVER_CHAT_PORTAL_TOOLS_HPP_
#define GUFO_SERVER_CHAT_PORTAL_TOOLS_HPP_

#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {

struct InbuiltToolDefinition {
  std::string name;
  std::string description;
  json::Value parameters;
  bool enabled_by_default{true};
  bool dangerous{false};
};

struct ToolExecutionResult {
  bool success{false};
  std::string output;
  std::string error;
};

class ToolEngine {
 public:
  static const std::vector<InbuiltToolDefinition>& GetInbuiltTools();
  static const InbuiltToolDefinition* FindInbuiltTool(const std::string& name);

  static ToolExecutionResult Execute(const std::string& name,
                                     const json::Value& arguments);

  static ToolExecutionResult ExecuteMcpProxy(const std::string& url,
                                            const std::string& headers_json,
                                            const std::string& tool_name,
                                            const json::Value& arguments);

  static ToolExecutionResult SyncMcpTools(const std::string& url,
                                          const std::string& headers_json);
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CHAT_PORTAL_TOOLS_HPP_
