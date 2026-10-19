#include "src/cli/serve/chat_portal_tools.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <regex>
#include <sstream>

namespace gufo::server {
namespace {

size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
  auto* str = static_cast<std::string*>(userp);
  const size_t total_size = size * nmemb;
  if (str->size() + total_size > 512 * 1024) {  // Cap at 512 KB
    const size_t allowed = 512 * 1024 - str->size();
    str->append(static_cast<char*>(contents), allowed);
    return total_size;  // Abort further writes or return size to avoid error
  }
  str->append(static_cast<char*>(contents), total_size);
  return total_size;
}

std::string HttpPost(const std::string& url, const std::string& payload,
                     const std::string& headers_json, std::string* error_out) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    if (error_out) *error_out = "Failed to initialize CURL";
    return "";
  }

  std::string response;
  struct curl_slist* chunk = nullptr;
  chunk = curl_slist_append(chunk, "Content-Type: application/json");
  chunk = curl_slist_append(chunk, "Accept: application/json");

  try {
    const auto hdrs = json::parse(headers_json);
    if (hdrs.is_object()) {
      for (const auto& kv : hdrs.members()) {
        std::string header_line = kv.first + ": " + kv.second.str();
        chunk = curl_slist_append(chunk, header_line.c_str());
      }
    }
  } catch (...) {}

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);  // Allow local test certs

  const CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    if (error_out) *error_out = curl_easy_strerror(res);
    response.clear();
  }

  curl_slist_free_all(chunk);
  curl_easy_cleanup(curl);
  return response;
}

std::string HttpGet(const std::string& url, std::string* error_out) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    if (error_out) *error_out = "Failed to initialize CURL";
    return "";
  }

  std::string response;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "Gufo-Chat/1.0 (AMD Strix Halo; gfx1151)");

  const CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    if (error_out) *error_out = curl_easy_strerror(res);
    response.clear();
  }

  curl_easy_cleanup(curl);
  return response;
}

std::string CurrentIsoTimeWithDay() {
  const auto now = std::chrono::system_clock::now();
  const auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm buf{};
#if defined(_WIN32)
  localtime_s(&buf, &in_time_t);
#else
  localtime_r(&in_time_t, &buf);
#endif

  std::ostringstream ss;
  ss << std::put_time(&buf, "%Y-%m-%dT%H:%M:%S");

  static const char* kDays[] = {"Sunday", "Monday", "Tuesday", "Wednesday",
                                "Thursday", "Friday", "Saturday"};
  const std::string day = (buf.tm_wday >= 0 && buf.tm_wday < 7) ? kDays[buf.tm_wday] : "";

  json::Value obj = json::Value::object();
  obj["datetime"] = ss.str();
  obj["day_of_week"] = day;
  obj["year"] = buf.tm_year + 1900;
  obj["month"] = buf.tm_mon + 1;
  obj["day"] = buf.tm_mday;
  return obj.dump();
}

bool WildcardMatch(std::string_view str, std::string_view pat) {
  size_t s = 0, p = 0, match = 0, star = std::string_view::npos;
  while (s < str.size()) {
    if (p < pat.size() && (pat[p] == '?' || pat[p] == str[s])) {
      s++;
      p++;
    } else if (p < pat.size() && pat[p] == '*') {
      star = p++;
      match = s;
    } else if (star != std::string_view::npos) {
      p = star + 1;
      s = ++match;
    } else {
      return false;
    }
  }
  while (p < pat.size() && pat[p] == '*') p++;
  return p == pat.size();
}

std::string ResolveTargetPath(const std::string& path_str,
                              const std::string& base_dir) {
  if (path_str.empty()) return "";
  std::filesystem::path p(path_str);
  if (base_dir.empty() || p.is_absolute()) {
    return path_str;
  }
  const auto combined = std::filesystem::path(base_dir) / p;
  return combined.lexically_normal().string();
}

}  // namespace

const std::vector<InbuiltToolDefinition>& ToolEngine::GetInbuiltTools() {
  static const std::vector<InbuiltToolDefinition> kTools = [] {
    std::vector<InbuiltToolDefinition> list;

    // 1. get_datetime
    {
      InbuiltToolDefinition tool;
      tool.name = "get_datetime";
      tool.description = "Get the current date, day of week, and local time in ISO-8601 format.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {},
        "required": []
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 2. read_file
    {
      InbuiltToolDefinition tool;
      tool.name = "read_file";
      tool.description = "Read text content from a file in the local filesystem.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Absolute or relative file path to read."
          },
          "offset": {
            "type": "integer",
            "description": "Optional 1-based start line number."
          },
          "limit": {
            "type": "integer",
            "description": "Optional maximum number of lines to read."
          }
        },
        "required": ["path"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 3. write_file
    {
      InbuiltToolDefinition tool;
      tool.name = "write_file";
      tool.description = "Create or overwrite a file with the provided text content.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "File path to create or overwrite."
          },
          "content": {
            "type": "string",
            "description": "The text content to write into the file."
          }
        },
        "required": ["path", "content"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 4. edit_file
    {
      InbuiltToolDefinition tool;
      tool.name = "edit_file";
      tool.description = "Edit an existing file by replacing a target string or block with replacement text.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Path to the file to edit."
          },
          "target": {
            "type": "string",
            "description": "The exact existing text block to be replaced."
          },
          "replacement": {
            "type": "string",
            "description": "The replacement text to insert in place of target."
          }
        },
        "required": ["path", "target", "replacement"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 5. apply_diff
    {
      InbuiltToolDefinition tool;
      tool.name = "apply_diff";
      tool.description = "Apply a unified diff patch to a target file.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Path of the file to patch."
          },
          "diff": {
            "type": "string",
            "description": "Unified diff content to apply."
          }
        },
        "required": ["path", "diff"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 6. file_glob_search
    {
      InbuiltToolDefinition tool;
      tool.name = "file_glob_search";
      tool.description = "Search for files matching a wildcard pattern (e.g. *.cpp, src/**/*.hpp).";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "Wildcard pattern or glob (e.g. *.py, test_*)."
          },
          "directory": {
            "type": "string",
            "description": "Directory to search from (defaults to '.')."
          }
        },
        "required": ["pattern"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 7. grep_search
    {
      InbuiltToolDefinition tool;
      tool.name = "grep_search";
      tool.description = "Search for a text pattern or regular expression across files in a path.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Text substring or regex pattern to match."
          },
          "path": {
            "type": "string",
            "description": "File or directory path to search within."
          },
          "case_insensitive": {
            "type": "boolean",
            "description": "Whether to perform a case-insensitive search (default false)."
          }
        },
        "required": ["query", "path"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    // 8. exec_shell_command
    {
      InbuiltToolDefinition tool;
      tool.name = "exec_shell_command";
      tool.description = "Execute a shell command on the host machine and capture stdout/stderr.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The shell command line string to execute."
          },
          "cwd": {
            "type": "string",
            "description": "Optional working directory in which to execute the command."
          }
        },
        "required": ["command"]
      })json");
      tool.enabled_by_default = false;  // Disabled by default for security confirmation
      tool.dangerous = true;
      list.push_back(std::move(tool));
    }

    // 9. fetch_web_page
    {
      InbuiltToolDefinition tool;
      tool.name = "fetch_web_page";
      tool.description = "Fetch the text or HTML content from a web URL using HTTP/HTTPS GET.";
      tool.parameters = json::parse(R"json({
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "description": "The full HTTP or HTTPS URL to fetch."
          }
        },
        "required": ["url"]
      })json");
      tool.enabled_by_default = true;
      tool.dangerous = false;
      list.push_back(std::move(tool));
    }

    return list;
  }();

  return kTools;
}

const InbuiltToolDefinition* ToolEngine::FindInbuiltTool(const std::string& name) {
  for (const auto& t : GetInbuiltTools()) {
    if (t.name == name) {
      return &t;
    }
  }
  return nullptr;
}

ToolExecutionResult ToolEngine::Execute(const std::string& name,
                                       const json::Value& arguments,
                                       const std::string& working_directory) {
  ToolExecutionResult res;

  if (name == "get_datetime") {
    res.success = true;
    res.output = CurrentIsoTimeWithDay();
    return res;
  }

  if (name == "read_file") {
    const std::string raw_path = arguments.member_str("path");
    if (raw_path.empty()) {
      res.error = "Missing required argument 'path'";
      return res;
    }
    const std::string path = ResolveTargetPath(raw_path, working_directory);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || std::filesystem::is_directory(path, ec)) {
      res.error = "File not found or is a directory: " + path;
      return res;
    }

    std::ifstream f(path);
    if (!f.is_open()) {
      res.error = "Failed to open file: " + path;
      return res;
    }

    const int offset = arguments.contains("offset") ? static_cast<int>(arguments.member_size("offset", 1)) : 1;
    const int limit = arguments.contains("limit") ? static_cast<int>(arguments.member_size("limit", 1000)) : 1000;

    std::string line;
    int current_line = 1;
    int lines_read = 0;
    std::ostringstream out;

    while (std::getline(f, line)) {
      if (current_line >= offset) {
        out << current_line << ": " << line << "\n";
        lines_read++;
        if (lines_read >= limit) break;
      }
      current_line++;
    }

    res.success = true;
    res.output = out.str();
    return res;
  }

  if (name == "write_file") {
    const std::string raw_path = arguments.member_str("path");
    const std::string content = arguments.member_str("content");
    if (raw_path.empty()) {
      res.error = "Missing required argument 'path'";
      return res;
    }

    const std::string path = ResolveTargetPath(raw_path, working_directory);
    const auto p = std::filesystem::path(path);
    if (p.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
      res.error = "Cannot write to file: " + path;
      return res;
    }

    f << content;
    res.success = true;
    res.output = "Successfully wrote " + std::to_string(content.size()) + " bytes to " + path;
    return res;
  }

  if (name == "edit_file") {
    const std::string raw_path = arguments.member_str("path");
    const std::string target = arguments.member_str("target");
    const std::string replacement = arguments.member_str("replacement");
    if (raw_path.empty() || target.empty()) {
      res.error = "Missing required 'path' or 'target' parameter";
      return res;
    }

    const std::string path = ResolveTargetPath(raw_path, working_directory);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      res.error = "Failed to open file for editing: " + path;
      return res;
    }

    std::ostringstream sstr;
    sstr << in.rdbuf();
    std::string content = sstr.str();
    in.close();

    const auto pos = content.find(target);
    if (pos == std::string::npos) {
      res.error = "Target string not found in file: " + path;
      return res;
    }

    content.replace(pos, target.length(), replacement);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      res.error = "Failed to save modified file: " + path;
      return res;
    }

    out << content;
    res.success = true;
    res.output = "Successfully edited " + path;
    return res;
  }

  if (name == "apply_diff") {
    const std::string raw_path = arguments.member_str("path");
    const std::string diff = arguments.member_str("diff");
    if (raw_path.empty() || diff.empty()) {
      res.error = "Missing required 'path' or 'diff' parameter";
      return res;
    }

    const std::string path = ResolveTargetPath(raw_path, working_directory);
    // Simple line-based unified diff applier
    std::ifstream in(path);
    if (!in.is_open()) {
      res.error = "Failed to open file: " + path;
      return res;
    }

    std::vector<std::string> lines;
    std::string l;
    while (std::getline(in, l)) lines.push_back(l);
    in.close();

    std::istringstream diff_stream(diff);
    std::string diff_line;
    std::vector<std::string> out_lines;
    size_t line_idx = 0;

    while (std::getline(diff_stream, diff_line)) {
      if (diff_line.starts_with("@@")) {
        // Hunk header
        continue;
      }
      if (diff_line.starts_with("+")) {
        out_lines.push_back(diff_line.substr(1));
      } else if (diff_line.starts_with("-")) {
        line_idx++;
      } else if (diff_line.starts_with(" ")) {
        out_lines.push_back(diff_line.substr(1));
        line_idx++;
      } else if (line_idx < lines.size()) {
        out_lines.push_back(lines[line_idx++]);
      }
    }
    while (line_idx < lines.size()) {
      out_lines.push_back(lines[line_idx++]);
    }

    std::ofstream out(path, std::ios::trunc);
    for (const auto& s : out_lines) out << s << "\n";
    res.success = true;
    res.output = "Successfully applied diff to " + path;
    return res;
  }

  if (name == "file_glob_search") {
    const std::string pattern = arguments.member_str("pattern");
    std::string dir = arguments.member_str("directory", "");
    if (pattern.empty()) {
      res.error = "Missing required argument 'pattern'";
      return res;
    }
    if (dir.empty() || dir == ".") {
      dir = working_directory.empty() ? "." : working_directory;
    } else {
      dir = ResolveTargetPath(dir, working_directory);
    }

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
      res.error = "Directory does not exist: " + dir;
      return res;
    }

    std::vector<std::string> matches;
    try {
      for (const auto& entry : std::filesystem::recursive_directory_iterator(
               dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
        if (entry.is_regular_file(ec)) {
          const auto filename = entry.path().filename().string();
          if (WildcardMatch(filename, pattern)) {
            matches.push_back(entry.path().string());
            if (matches.size() >= 100) break;
          }
        }
      }
    } catch (...) {}

    json::Value arr = json::Value::array();
    for (const auto& m : matches) arr.push_back(m);
    res.success = true;
    res.output = arr.dump();
    return res;
  }

  if (name == "grep_search") {
    const std::string query = arguments.member_str("query");
    std::string raw_path = arguments.member_str("path", "");
    const bool case_insensitive = arguments.contains("case_insensitive") && arguments.find("case_insensitive")->as_bool(false);

    if (query.empty() || raw_path.empty()) {
      res.error = "Missing required 'query' or 'path' parameter";
      return res;
    }

    if (raw_path == "." || raw_path.empty()) {
      raw_path = working_directory.empty() ? "." : working_directory;
    } else {
      raw_path = ResolveTargetPath(raw_path, working_directory);
    }
    const std::string path = raw_path;

    std::error_code ec;
    std::vector<std::filesystem::path> files_to_search;
    if (std::filesystem::is_directory(path, ec)) {
      for (const auto& entry : std::filesystem::recursive_directory_iterator(
               path, std::filesystem::directory_options::skip_permission_denied, ec)) {
        if (entry.is_regular_file(ec)) {
          files_to_search.push_back(entry.path());
          if (files_to_search.size() >= 50) break;
        }
      }
    } else if (std::filesystem::is_regular_file(path, ec)) {
      files_to_search.push_back(path);
    } else {
      res.error = "Path does not exist: " + path;
      return res;
    }

    std::string query_low = query;
    if (case_insensitive) {
      std::transform(query_low.begin(), query_low.end(), query_low.begin(), ::tolower);
    }

    std::ostringstream out;
    int matches_found = 0;

    for (const auto& fpath : files_to_search) {
      std::ifstream fin(fpath);
      if (!fin.is_open()) continue;

      std::string line;
      int line_no = 1;
      while (std::getline(fin, line)) {
        std::string check = line;
        if (case_insensitive) {
          std::transform(check.begin(), check.end(), check.begin(), ::tolower);
        }
        if (check.find(query_low) != std::string::npos) {
          out << fpath.string() << ":" << line_no << ": " << line << "\n";
          matches_found++;
          if (matches_found >= 100) break;
        }
        line_no++;
      }
      if (matches_found >= 100) break;
    }

    res.success = true;
    res.output = out.str().empty() ? "No matches found." : out.str();
    return res;
  }

  if (name == "exec_shell_command") {
    const std::string command = arguments.member_str("command");
    std::string cwd = arguments.member_str("cwd", "");
    if (command.empty()) {
      res.error = "Missing required argument 'command'";
      return res;
    }

    if (cwd.empty() && !working_directory.empty()) {
      cwd = working_directory;
    } else if (!cwd.empty() && !working_directory.empty()) {
      cwd = ResolveTargetPath(cwd, working_directory);
    }

    std::string full_cmd;
#if defined(_WIN32)
    full_cmd = "cmd.exe /c \"";
    if (!cwd.empty()) {
      full_cmd += "cd /D \"" + cwd + "\" && ";
    }
    full_cmd += command + "\" 2>&1";
    FILE* pipe = _popen(full_cmd.c_str(), "r");
#else
    full_cmd = "";
    if (!cwd.empty()) {
      full_cmd += "cd \"" + cwd + "\" && ";
    }
    full_cmd += command + " 2>&1";
    FILE* pipe = popen(full_cmd.c_str(), "r");
#endif

    if (!pipe) {
      res.error = "Failed to launch shell process";
      return res;
    }

    std::array<char, 2048> buffer;
    std::string output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
      output.append(buffer.data());
      if (output.size() > 64 * 1024) {  // 64 KB cap
        output += "\n[output truncated at 64 KB]";
        break;
      }
    }

#if defined(_WIN32)
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif

    res.success = (status == 0);
    res.output = output.empty() ? "(command completed with no output)" : output;
    if (status != 0) {
      res.error = "Exit code " + std::to_string(status);
    }
    return res;
  }

  if (name == "fetch_web_page") {
    const std::string url = arguments.member_str("url");
    if (url.empty()) {
      res.error = "Missing required argument 'url'";
      return res;
    }

    std::string err;
    const std::string content = HttpGet(url, &err);
    if (!err.empty()) {
      res.error = "HTTP request failed: " + err;
      return res;
    }

    res.success = true;
    res.output = content;
    return res;
  }

  res.error = "Unknown inbuilt tool: " + name;
  return res;
}

ToolExecutionResult ToolEngine::ExecuteMcpProxy(const std::string& url,
                                               const std::string& headers_json,
                                               const std::string& tool_name,
                                               const json::Value& arguments) {
  ToolExecutionResult res;
  json::Value rpc = json::Value::object();
  rpc["jsonrpc"] = "2.0";
  rpc["id"] = 1;
  rpc["method"] = "tools/call";
  json::Value params = json::Value::object();
  params["name"] = tool_name;
  params["arguments"] = arguments;
  rpc["params"] = std::move(params);

  std::string err;
  const std::string resp_str = HttpPost(url, rpc.dump(), headers_json, &err);
  if (!err.empty()) {
    res.error = "MCP server call failed: " + err;
    return res;
  }

  try {
    const auto parsed = json::parse(resp_str);
    if (parsed.contains("error")) {
      res.error = parsed.member_str("error");
      return res;
    }
    if (const auto* r = parsed.find("result")) {
      if (const auto* content = r->find("content"); content && content->is_array()) {
        std::ostringstream out;
        for (const auto& item : content->items()) {
          if (item.contains("text")) {
            out << item.member_str("text");
          }
        }
        res.success = true;
        res.output = out.str();
        return res;
      }
      res.success = true;
      res.output = r->dump();
      return res;
    }
    res.success = true;
    res.output = resp_str;
  } catch (...) {
    res.success = true;
    res.output = resp_str;
  }
  return res;
}

ToolExecutionResult ToolEngine::SyncMcpTools(const std::string& url,
                                             const std::string& headers_json) {
  ToolExecutionResult res;
  json::Value rpc = json::Value::object();
  rpc["jsonrpc"] = "2.0";
  rpc["id"] = 1;
  rpc["method"] = "tools/list";
  rpc["params"] = json::Value::object();

  std::string err;
  const std::string resp_str = HttpPost(url, rpc.dump(), headers_json, &err);
  if (!err.empty()) {
    res.error = "Failed to query MCP tools/list: " + err;
    return res;
  }

  try {
    const auto parsed = json::parse(resp_str);
    if (const auto* r = parsed.find("result")) {
      if (const auto* tools = r->find("tools"); tools) {
        res.success = true;
        res.output = tools->dump();
        return res;
      }
    }
    res.success = true;
    res.output = resp_str;
  } catch (...) {
    res.error = "Failed to parse JSON response from MCP server: " + resp_str;
  }
  return res;
}

}  // namespace gufo::server
