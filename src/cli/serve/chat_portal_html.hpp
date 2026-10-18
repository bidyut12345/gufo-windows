#ifndef GUFO_SERVER_CHAT_PORTAL_HTML_HPP_
#define GUFO_SERVER_CHAT_PORTAL_HTML_HPP_

#include <string_view>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
#endif

namespace gufo::server {

inline constexpr std::string_view kChatPortalHtml = R"rawhtml(<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Gufo Chat</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#x1F989;</text></svg>">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #09090b;
      --bg-sidebar: #121215;
      --bg-card: #18181b;
      --bg-elevated: #27272a;
      --bg-hover: #323238;
      --border: #27272a;
      --border-strong: #3f3f46;
      --text: #f4f4f5;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --primary: #fafafa;
      --primary-text: #09090b;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #22c55e;
      --danger: #ef4444;
      --radius-sm: 4px;
      --radius: 8px;
      --radius-lg: 12px;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background-color: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Navigation Bar - llama.cpp style */
    header.top-nav {
      height: 52px;
      background-color: var(--bg-sidebar);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      flex-shrink: 0;
      z-index: 30;
    }

    .nav-left {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .brand-llama {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 1.05rem;
      letter-spacing: -0.02em;
      color: var(--text);
      text-decoration: none;
      user-select: none;
    }
    .brand-icon {
      font-size: 1.3rem;
      line-height: 1;
    }
    .brand-badge {
      background: var(--bg-elevated);
      color: var(--text-muted);
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-strong);
      font-family: var(--font-mono);
      font-weight: 500;
    }

    /* Tabs (Chat vs Completion) */
    .nav-tabs {
      display: flex;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2px;
      margin-left: 8px;
    }
    .tab-item {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 4px 14px;
      font-size: 0.82rem;
      font-weight: 500;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s;
    }
    .tab-item.active {
      background: var(--bg-card);
      color: var(--text);
      font-weight: 600;
      box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }
    .tab-item:hover:not(.active) {
      color: var(--text);
    }

    .nav-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .slot-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(34, 197, 94, 0.1);
      border: 1px solid rgba(34, 197, 94, 0.25);
      color: #86efac;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      padding: 4px 10px;
      border-radius: 20px;
    }
    .slot-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 6px var(--success);
    }

    .model-pill {
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.78rem;
      padding: 4px 10px;
      border-radius: var(--radius);
      max-width: 220px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: var(--font-mono);
    }

    .btn-top {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius);
      padding: 5px 10px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s;
    }
    .btn-top:hover {
      background: var(--bg-card);
      color: var(--text);
      border-color: var(--border-strong);
    }

    /* Main Workspace */
    .main-workspace {
      display: flex;
      flex: 1;
      height: calc(100vh - 52px);
      overflow: hidden;
      position: relative;
    }

    /* Left Sidebar */
    aside#sidebar {
      width: 260px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      z-index: 20;
    }

    .sidebar-top {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      border-bottom: 1px solid var(--border);
    }

    .btn-new-chat {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      color: var(--text);
      border-radius: var(--radius);
      padding: 8px 12px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background 0.15s, border-color 0.15s;
    }
    .btn-new-chat:hover {
      background: var(--bg-elevated);
      border-color: #52525b;
    }

    .search-input {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: var(--radius);
      padding: 6px 10px;
      font-size: 0.8rem;
      outline: none;
      width: 100%;
    }
    .search-input:focus {
      border-color: var(--border-strong);
    }

    .sidebar-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .chat-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 10px;
      border-radius: var(--radius);
      color: var(--text-muted);
      font-size: 0.85rem;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s;
    }
    .chat-item:hover {
      background: var(--bg-card);
      color: var(--text);
    }
    .chat-item.active {
      background: var(--bg-elevated);
      color: var(--text);
      font-weight: 500;
    }
    .chat-title-text {
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-right: 6px;
    }
    .chat-item-actions {
      display: none;
      gap: 4px;
    }
    .chat-item:hover .chat-item-actions, .chat-item.active .chat-item-actions {
      display: flex;
    }
    .btn-mini {
      background: transparent;
      border: none;
      color: var(--text-dim);
      padding: 2px;
      cursor: pointer;
      border-radius: var(--radius-sm);
    }
    .btn-mini:hover { color: var(--text); }

    .sidebar-bottom {
      padding: 10px 14px;
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(18, 18, 21, 0.95);
      font-size: 0.8rem;
    }
    .user-info {
      display: flex;
      align-items: center;
      gap: 8px;
      overflow: hidden;
    }
    .user-tag {
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .sync-status-badge {
      font-size: 0.68rem;
      color: var(--success);
      display: flex;
      align-items: center;
      gap: 4px;
    }

    /* Content Area */
    .content-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      position: relative;
      background: var(--bg);
      overflow: hidden;
    }

    /* Chat View */
    .chat-view {
      display: flex;
      flex-direction: column;
      flex: 1;
      height: 100%;
      overflow: hidden;
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px 20px 32px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
    }

    /* Empty state */
    .llama-hero {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      margin: auto 0;
      text-align: center;
      padding: 40px 16px;
    }
    .llama-hero-icon {
      font-size: 3.5rem;
      margin-bottom: 12px;
      line-height: 1;
    }
    .llama-hero-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 6px;
      letter-spacing: -0.02em;
    }
    .llama-hero-subtitle {
      color: var(--text-muted);
      font-size: 0.9rem;
      max-width: 480px;
      margin-bottom: 24px;
      line-height: 1.5;
    }

    /* Message Bubbles - llama.cpp format */
    .msg-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
      width: 100%;
    }
    .msg-author {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 2px;
      user-select: none;
    }
    .msg-author.user { color: #93c5fd; }
    .msg-author.assistant { color: #f4f4f5; }

    .msg-body {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 16px;
      font-size: 0.93rem;
      line-height: 1.6;
      position: relative;
    }
    .msg-group.user .msg-body {
      background: #181c24;
      border-color: #27303f;
    }

    /* Collapsible Thinking Process - DeepSeek / llama-server style */
    .thought-accordion {
      margin-bottom: 12px;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: rgba(24, 24, 27, 0.6);
      overflow: hidden;
      font-size: 0.85rem;
    }
    .thought-summary {
      padding: 8px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--text-muted);
      font-size: 0.8rem;
      font-family: var(--font-mono);
      cursor: pointer;
      user-select: none;
      background: rgba(39, 39, 42, 0.5);
    }
    .thought-summary:hover {
      background: var(--bg-elevated);
      color: var(--text);
    }
    .thought-text {
      padding: 10px 14px;
      color: var(--text-dim);
      font-size: 0.84rem;
      line-height: 1.5;
      white-space: pre-wrap;
      border-top: 1px solid var(--border);
      background: rgba(9, 9, 11, 0.5);
    }
    .thought-text.collapsed {
      display: none;
    }

    /* Markdown inside messages */
    .md-content p { margin-bottom: 8px; }
    .md-content p:last-child { margin-bottom: 0; }
    .md-content pre {
      background: #000;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      margin: 10px 0;
      overflow-x: auto;
    }
    .code-topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 12px;
      background: #121215;
      border-bottom: 1px solid var(--border);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--text-dim);
    }
    .btn-copy {
      background: transparent;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 0.72rem;
    }
    .btn-copy:hover { color: var(--text); }
    .md-content code {
      font-family: var(--font-mono);
      font-size: 0.86em;
    }
    .md-content pre code {
      display: block;
      padding: 12px;
      color: #e4e4e7;
      line-height: 1.45;
    }
    .md-content :not(pre) > code {
      background: var(--bg-elevated);
      padding: 2px 5px;
      border-radius: 4px;
      color: #e4e4e7;
    }
    .md-content table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 0.86rem;
    }
    .md-content th, .md-content td {
      border: 1px solid var(--border);
      padding: 6px 10px;
      text-align: left;
    }
    .md-content th { background: var(--bg-elevated); }

    /* Message Actions */
    .msg-actions {
      display: flex;
      gap: 6px;
      margin-top: 6px;
    }
    .btn-msg-action {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 0.72rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 2px 6px;
      border-radius: var(--radius-sm);
    }
    .btn-msg-action:hover {
      background: var(--bg-card);
      color: var(--text-muted);
    }

    /* Chat Input Area */
    .chat-bottom {
      padding: 0 16px 16px;
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .input-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .input-box:focus-within {
      border-color: var(--border-strong);
    }

    .image-preview-bar {
      display: none;
      align-items: center;
      gap: 8px;
      padding-bottom: 4px;
    }
    .image-thumb {
      width: 50px;
      height: 50px;
      border-radius: var(--radius-sm);
      object-fit: cover;
      border: 1px solid var(--border);
    }
    .btn-remove-thumb {
      background: rgba(0,0,0,0.6);
      border: none;
      color: var(--text);
      cursor: pointer;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      font-size: 11px;
    }

    .input-row {
      display: flex;
      align-items: flex-end;
      gap: 8px;
    }

    textarea#chat-input {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text);
      font-family: inherit;
      font-size: 0.92rem;
      resize: none;
      min-height: 24px;
      max-height: 160px;
      outline: none;
      line-height: 1.5;
    }

    .btn-icon-input {
      background: transparent;
      border: none;
      color: var(--text-dim);
      cursor: pointer;
      width: 32px;
      height: 32px;
      border-radius: var(--radius);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .btn-icon-input:hover {
      color: var(--text);
      background: var(--bg-elevated);
    }

    .btn-submit {
      background: var(--primary);
      color: var(--primary-text);
      border: none;
      border-radius: var(--radius);
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
      transition: opacity 0.15s;
    }
    .btn-submit:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    .btn-stop-stream {
      display: none;
      background: var(--danger);
      color: white;
      border: none;
      border-radius: var(--radius);
      width: 32px;
      height: 32px;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
    }

    .chat-stats-footer {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      padding: 0 4px;
    }

    /* Completion View */
    .completion-view {
      display: none;
      flex: 1;
      flex-direction: column;
      padding: 16px;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
      gap: 12px;
      overflow-y: auto;
    }
    .completion-textarea {
      width: 100%;
      flex: 1;
      min-height: 200px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 0.88rem;
      resize: vertical;
      outline: none;
    }
    .completion-textarea:focus { border-color: var(--border-strong); }
    .completion-actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .btn-completion-run {
      background: var(--primary);
      color: var(--primary-text);
      border: none;
      border-radius: var(--radius);
      padding: 8px 18px;
      font-weight: 600;
      cursor: pointer;
    }

    /* Settings Drawer */
    .drawer-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(4px);
      z-index: 50;
    }
    .drawer-backdrop.show { display: block; }

    .drawer-panel {
      position: fixed;
      right: 0;
      top: 0;
      bottom: 0;
      width: 420px;
      max-width: 90vw;
      background: var(--bg-sidebar);
      border-left: 1px solid var(--border);
      z-index: 60;
      display: flex;
      flex-direction: column;
      transform: translateX(100%);
      transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .drawer-panel.show { transform: translateX(0); }

    .drawer-header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .drawer-title { font-size: 0.95rem; font-weight: 700; }
    .drawer-body {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .setting-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .setting-label {
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    .setting-desc {
      font-size: 0.72rem;
      color: var(--text-dim);
    }
    .setting-input, .setting-textarea {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 10px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.85rem;
      outline: none;
    }
    .setting-input:focus, .setting-textarea:focus { border-color: var(--border-strong); }
    .setting-range-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .setting-range {
      flex: 1;
      accent-color: var(--text);
    }
    .setting-val {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      min-width: 44px;
      text-align: right;
      color: var(--text);
    }

    /* Auth Dialog / Modal */
    .dialog-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(6px);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }
    .dialog-backdrop.show { display: flex; }
    .dialog-box {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      width: 100%;
      max-width: 380px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }
    .dialog-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .dialog-body {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .btn-dialog-submit {
      background: var(--primary);
      color: var(--primary-text);
      border: none;
      border-radius: var(--radius);
      padding: 10px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
    }
    .dialog-alert {
      padding: 8px 12px;
      border-radius: var(--radius);
      font-size: 0.8rem;
      display: none;
    }
    .dialog-alert.error {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
    }

    @media (max-width: 768px) {
      aside#sidebar {
        position: absolute;
        left: 0; top: 0; bottom: 0;
        transform: translateX(-100%);
      }
      aside#sidebar.open { transform: translateX(0); }
      .nav-tabs { display: none; }
    }
    /* Tools & MCP Styles */
    .tools-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 20px;
      font-size: 0.75rem;
      cursor: pointer;
      color: var(--text-muted);
      transition: all 0.15s ease;
    }
    .tools-pill:hover {
      background: var(--bg-hover);
      color: var(--text);
      border-color: var(--border-strong);
    }
    .tools-pill.active {
      border-color: #3b82f6;
      color: #93c5fd;
      background: rgba(59, 130, 246, 0.12);
    }
    .tools-pill-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #3b82f6;
    }
    .tools-dialog-box {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      width: 90%;
      max-width: 720px;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      overflow: hidden;
    }
    .tools-tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: var(--bg-sidebar);
      padding: 0 16px;
      gap: 12px;
    }
    .tools-tab-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 0.82rem;
      font-weight: 500;
      padding: 10px 4px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
    }
    .tools-tab-btn.active {
      color: var(--text);
      border-bottom-color: var(--accent);
    }
    .tools-content-pane {
      padding: 16px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .tool-item-card {
      background: var(--bg-sidebar);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .tool-item-info { flex: 1; }
    .tool-item-name {
      font-weight: 600;
      font-size: 0.88rem;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .tool-item-desc {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 4px;
      line-height: 1.4;
    }
    .tool-danger-badge {
      font-size: 0.68rem;
      background: rgba(239, 68, 68, 0.2);
      color: #fca5a5;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 500;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .tool-mcp-badge {
      font-size: 0.68rem;
      background: rgba(168, 85, 247, 0.2);
      color: #d8b4fe;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 500;
      border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .switch-toggle {
      position: relative;
      display: inline-block;
      width: 36px;
      height: 20px;
      flex-shrink: 0;
    }
    .switch-toggle input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: var(--bg-hover);
      border-radius: 20px;
      transition: .2s;
      border: 1px solid var(--border);
    }
    .slider:before {
      position: absolute;
      content: "";
      height: 14px;
      width: 14px;
      left: 2px;
      bottom: 2px;
      background-color: white;
      border-radius: 50%;
      transition: .2s;
    }
    input:checked + .slider {
      background-color: var(--accent);
      border-color: var(--accent);
    }
    input:checked + .slider:before {
      transform: translateX(16px);
    }
    .tool-exec-card {
      background: #111114;
      border: 1px solid #2e2e38;
      border-left: 3px solid #3b82f6;
      border-radius: var(--radius);
      padding: 10px 12px;
      margin: 8px 0;
      font-size: 0.8rem;
      font-family: var(--font-mono);
    }
    .tool-exec-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      font-weight: 600;
      color: #93c5fd;
    }
    .tool-exec-status {
      font-size: 0.7rem;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 500;
    }
    .tool-exec-status.running {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      animation: pulse 1.5s infinite;
    }
    .tool-exec-status.success {
      background: rgba(34, 197, 94, 0.2);
      color: #4ade80;
    }
    .tool-exec-status.error {
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    .tool-exec-body {
      margin-top: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .tool-exec-body.collapsed {
      display: none;
    }
    .tool-exec-section-label {
      font-size: 0.7rem;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
    }
    .tool-exec-box {
      background: #09090b;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 6px 8px;
      max-height: 200px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
      color: #e4e4e7;
    }
  </style>
</head>
<body>
  <!-- Top Navigation Bar -->
  <header class="top-nav">
    <div class="nav-left">
      <button id="btn-toggle-sidebar" class="btn-top" title="Toggle Sidebar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
      </button>
      <a href="#" class="brand-llama">
        <span class="brand-icon">&#x1F989;</span>
        <span>Gufo Chat</span>
      </a>
      <span class="brand-badge">server</span>
      <div class="nav-tabs">
        <button id="tab-nav-chat" class="tab-item active">Chat</button>
        <button id="tab-nav-comp" class="tab-item">Completion</button>
      </div>
    </div>

    <div class="nav-right">
      <div id="slot-indicator" class="slot-pill" title="Server Slots Status">
        <span class="slot-dot"></span>
        <span id="slot-text">1/1 slots (idle)</span>
      </div>
      <div id="model-pill" class="model-pill" title="Loaded Model">loading model...</div>
      <button id="btn-open-tools" class="btn-top" title="Tools & MCP Servers">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>
        <span id="nav-tools-label">Tools & MCP</span>
      </button>
      <button id="btn-open-drawer" class="btn-top" title="Parameters & Settings">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        <span>Settings</span>
      </button>
      <button id="btn-auth-user" class="btn-top" title="Account & Multi-Device Sync">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
        <span id="auth-btn-label">Sign In</span>
      </button>
    </div>
  </header>

  <!-- Workspace -->
  <div class="main-workspace">
    <!-- Left Sidebar -->
    <aside id="sidebar">
      <div class="sidebar-top">
        <button id="btn-new-chat" class="btn-new-chat">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          <span>New chat</span>
        </button>
        <input type="text" id="conv-search-input" class="search-input" placeholder="Search chats...">
      </div>

      <div id="chat-list" class="sidebar-list">
        <!-- Injected dynamically -->
      </div>

      <div class="sidebar-bottom">
        <div class="user-info">
          <span id="sidebar-user-name" class="user-tag">Guest</span>
        </div>
        <div class="sync-status-badge">
          <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--success);"></span>
          <span id="sync-label">Synced DB</span>
        </div>
      </div>
    </aside>

    <!-- Main Viewport -->
    <main class="content-area">
      <!-- Chat Interface -->
      <div id="chat-interface" class="chat-view">
        <div id="messages-container" class="chat-messages">
          <!-- Rendered dynamically -->
        </div>

        <div class="chat-bottom">
          <div class="input-box">
            <div id="image-bar" class="image-preview-bar">
              <img id="thumb-img" class="image-thumb" src="" alt="preview">
              <button id="btn-remove-thumb" class="btn-remove-thumb">&times;</button>
            </div>
            <div class="input-row">
              <input type="file" id="file-uploader" accept="image/*" style="display:none">
              <button id="btn-upload-file" class="btn-icon-input" title="Attach file / image">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
              </button>
              <button id="btn-quick-tools" class="tools-pill active" title="Toggle tools or configure MCP" style="flex-shrink:0;">
                <span class="tools-pill-dot"></span>
                <span id="quick-tools-label">Tools: 8</span>
              </button>
              <textarea id="chat-input" rows="1" placeholder="Say something... (Shift+Enter for newline)"></textarea>
              <button id="btn-stop" class="btn-stop-stream" title="Stop generating">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg>
              </button>
              <button id="btn-send" class="btn-submit" title="Send message">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
              </button>
            </div>
          </div>
          <div class="chat-stats-footer">
            <span id="footer-speed-metrics">eval: 0.0 t/s | prompt: 0.0 t/s</span>
            <span>Gufo Chat gfx1151</span>
          </div>
        </div>
      </div>

      <!-- Completion Interface -->
      <div id="completion-interface" class="completion-view">
        <div style="font-weight:600; font-size:0.95rem; margin-bottom:4px;">Raw Text Completion</div>
        <textarea id="comp-input" class="completion-textarea" placeholder="Prompt text to continue..."></textarea>
        <div class="completion-actions">
          <button id="btn-comp-generate" class="btn-completion-run">Generate</button>
          <button id="btn-comp-clear" class="btn-top">Clear</button>
          <span id="comp-status" style="font-size:0.8rem; color:var(--text-dim); font-family:var(--font-mono);"></span>
        </div>
      </div>
    </main>
  </div>

  <!-- Settings Drawer (llama.cpp style) -->
  <div id="drawer-backdrop" class="drawer-backdrop"></div>
  <aside id="drawer-panel" class="drawer-panel">
    <div class="drawer-header">
      <span class="drawer-title">Sampling & Server Settings</span>
      <button id="btn-close-drawer" class="btn-top" style="padding:2px 8px;">&times;</button>
    </div>
    <div class="drawer-body">
      <div class="setting-group">
        <label class="setting-label">System Prompt</label>
        <textarea id="set-system-prompt" class="setting-textarea" rows="3"></textarea>
        <span class="setting-desc">Sets default system context prepended to conversations.</span>
      </div>

      <div class="setting-group">
        <div class="setting-label">
          <span>Temperature</span>
          <span id="val-temp" class="setting-val">0.60</span>
        </div>
        <div class="setting-range-row">
          <input type="range" id="range-temp" class="setting-range" min="0" max="2.0" step="0.05" value="0.6">
        </div>
        <span class="setting-desc">Adjust randomness of generated tokens.</span>
      </div>

      <div class="setting-group">
        <div class="setting-label">
          <span>Top P</span>
          <span id="val-top-p" class="setting-val">0.95</span>
        </div>
        <div class="setting-range-row">
          <input type="range" id="range-top-p" class="setting-range" min="0.1" max="1.0" step="0.05" value="0.95">
        </div>
        <span class="setting-desc">Nucleus sampling threshold.</span>
      </div>

      <div class="setting-group">
        <div class="setting-label">
          <span>Max Tokens (n_predict)</span>
        </div>
        <input type="number" id="inp-max-tokens" class="setting-input" value="4096" min="64" max="32768" step="128">
      </div>

      <div class="setting-group" style="margin-top:10px;">
        <div style="font-weight:600; font-size:0.85rem; color:var(--text); margin-bottom:4px;">Server & Storage</div>
        <div style="font-size:0.75rem; color:var(--text-dim); line-height:1.4;">
          Database location: <code style="color:var(--text-muted);">%LOCALAPPDATA%\gufo\chat_portal.db</code><br>
          Acceleration: <span style="color:#86efac;">AMD Strix Halo gfx1151</span>
        </div>
      </div>

      <button id="btn-save-drawer" class="btn-dialog-submit" style="margin-top:10px;">Save Settings</button>
    </div>
  </aside>

  <!-- Auth Modal (Sign In / Register) -->
  <div id="auth-dialog" class="dialog-backdrop">
    <div class="dialog-box">
      <div class="dialog-header">
        <span id="auth-title" style="font-weight:700; font-size:1rem;">Sign In</span>
        <button id="btn-close-auth" class="btn-top" style="padding:2px 8px;">&times;</button>
      </div>
      <div class="dialog-body">
        <div id="auth-err" class="dialog-alert error"></div>
        <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
          Sign in or create an account to persist and synchronize your chat history across any device on your network.
        </div>
        <div class="setting-group">
          <label class="setting-label">Username</label>
          <input type="text" id="auth-user-inp" class="setting-input" placeholder="e.g. gufo_user">
        </div>
        <div class="setting-group">
          <label class="setting-label">Password</label>
          <input type="password" id="auth-pass-inp" class="setting-input" placeholder="••••••••">
        </div>
        <button id="btn-auth-exec" class="btn-dialog-submit">Sign In</button>
        <div style="text-align:center; font-size:0.8rem; color:var(--text-dim);">
          <a href="#" id="link-toggle-mode" style="color:var(--text-muted); text-decoration:underline;">Don't have an account? Create one</a>
        </div>
      </div>
    </div>
  </div>

  <!-- Tools & MCP Modal -->
  <div id="tools-dialog" class="dialog-backdrop">
    <div class="tools-dialog-box">
      <div class="dialog-header">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:1.1rem;">🔧</span>
          <span style="font-weight:700; font-size:1rem;">Tools & Model Context Protocol (MCP)</span>
        </div>
        <button id="btn-close-tools" class="btn-top" style="padding:2px 8px;">&times;</button>
      </div>

      <div class="tools-tabs">
        <button id="tab-inbuilt-tools" class="tools-tab-btn active">Inbuilt Tools (llama.cpp)</button>
        <button id="tab-mcp-servers" class="tools-tab-btn">MCP Servers</button>
        <button id="tab-custom-tools" class="tools-tab-btn">Custom Tools</button>
      </div>

      <!-- Inbuilt Tools Pane -->
      <div id="pane-inbuilt-tools" class="tools-content-pane">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:0.8rem; color:var(--text-muted);">Standard built-in system tools executed securely on the host server.</span>
          <div style="display:flex; gap:6px;">
            <button id="btn-enable-all-tools" class="btn-top" style="font-size:0.75rem; padding:3px 8px;">Enable All</button>
            <button id="btn-disable-all-tools" class="btn-top" style="font-size:0.75rem; padding:3px 8px;">Disable All</button>
          </div>
        </div>
        <div id="inbuilt-tools-list" style="display:flex; flex-direction:column; gap:8px;"></div>
      </div>

      <!-- MCP Servers Pane -->
      <div id="pane-mcp-servers" class="tools-content-pane" style="display:none;">
        <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
          Connect external Model Context Protocol (MCP) servers via Server-Sent Events (SSE) or HTTP stream. Tools provided by these servers are automatically exposed to Gufo Chat.
        </div>
        
        <div style="background:var(--bg-elevated); padding:12px; border-radius:var(--radius); display:flex; flex-direction:column; gap:10px;">
          <div style="font-weight:600; font-size:0.84rem;">Add MCP Server</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <input type="text" id="mcp-srv-name" class="setting-input" placeholder="Server Name (e.g. Memory, GitHub)">
            <input type="text" id="mcp-srv-url" class="setting-input" placeholder="URL (e.g. http://localhost:8001/sse)">
          </div>
          <div style="display:grid; grid-template-columns:120px 1fr; gap:8px;">
            <select id="mcp-srv-transport" class="setting-input">
              <option value="sse">SSE</option>
              <option value="http">HTTP</option>
            </select>
            <input type="text" id="mcp-srv-headers" class="setting-input" placeholder='Custom Headers JSON (e.g. {"Authorization":"Bearer ..."})'>
          </div>
          <button id="btn-add-mcp-server" class="btn-dialog-submit" style="padding:7px; font-size:0.82rem;">Add & Discover Tools</button>
        </div>

        <div style="font-weight:600; font-size:0.84rem; margin-top:6px;">Configured MCP Servers</div>
        <div id="mcp-servers-list" style="display:flex; flex-direction:column; gap:8px;"></div>
      </div>

      <!-- Custom Tools Pane -->
      <div id="pane-custom-tools" class="tools-content-pane" style="display:none;">
        <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
          Define custom function tools with JSON Schema parameters. When called by the model, Gufo can invoke an HTTP handler URL or return the arguments.
        </div>

        <div style="background:var(--bg-elevated); padding:12px; border-radius:var(--radius); display:flex; flex-direction:column; gap:10px;">
          <div style="font-weight:600; font-size:0.84rem;">Add Custom Function Tool</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <input type="text" id="custom-tool-name" class="setting-input" placeholder="Function Name (e.g. query_db)">
            <input type="text" id="custom-tool-handler" class="setting-input" placeholder="Handler URL (optional endpoint)">
          </div>
          <input type="text" id="custom-tool-desc" class="setting-input" placeholder="Description of what this tool does...">
          <div>
            <div style="font-size:0.75rem; color:var(--text-dim); margin-bottom:3px;">Parameters (JSON Schema):</div>
            <textarea id="custom-tool-schema" class="setting-input" style="height:70px; font-family:var(--font-mono); font-size:0.75rem;" placeholder='{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}'></textarea>
          </div>
          <button id="btn-add-custom-tool" class="btn-dialog-submit" style="padding:7px; font-size:0.82rem;">Add Custom Tool</button>
        </div>

        <div style="font-weight:600; font-size:0.84rem; margin-top:6px;">Your Custom Tools</div>
        <div id="custom-tools-list" style="display:flex; flex-direction:column; gap:8px;"></div>
      </div>
    </div>
  </div>

  <script>
    // State management
    const APP = {
      token: localStorage.getItem('gufo_auth_token') || '',
      user: null,
      conversations: [],
      activeId: null,
      activeConv: null,
      attachedImage: null,
      isGenerating: false,
      abortController: null,
      mode: 'chat', // 'chat' or 'comp'
      settings: {
        system_prompt: 'You are a helpful, respectful and honest assistant.',
        temperature: 0.6,
        top_p: 0.95,
        max_tokens: 4096
      },
      tools: {
        inbuilt: [],
        mcpServers: [],
        customTools: [],
        activeTab: 'inbuilt'
      }
    };

    // DOM references
    const $ = id => document.getElementById(id);
    const UI = {
      tabNavChat: $('tab-nav-chat'),
      tabNavComp: $('tab-nav-comp'),
      chatInterface: $('chat-interface'),
      completionInterface: $('completion-interface'),
      slotText: $('slot-text'),
      modelPill: $('model-pill'),
      btnToggleSidebar: $('btn-toggle-sidebar'),
      sidebar: $('sidebar'),
      btnNewChat: $('btn-new-chat'),
      convSearchInput: $('conv-search-input'),
      chatList: $('chat-list'),
      sidebarUserName: $('sidebar-user-name'),
      authBtnLabel: $('auth-btn-label'),
      btnAuthUser: $('btn-auth-user'),
      messagesContainer: $('messages-container'),
      imageBar: $('image-bar'),
      thumbImg: $('thumb-img'),
      btnRemoveThumb: $('btn-remove-thumb'),
      fileUploader: $('file-uploader'),
      btnUploadFile: $('btn-upload-file'),
      chatInput: $('chat-input'),
      btnSend: $('btn-send'),
      btnStop: $('btn-stop'),
      footerSpeedMetrics: $('footer-speed-metrics'),
      // Completion
      compInput: $('comp-input'),
      btnCompGenerate: $('btn-comp-generate'),
      btnCompClear: $('btn-comp-clear'),
      compStatus: $('comp-status'),
      // Drawer
      btnOpenDrawer: $('btn-open-drawer'),
      btnCloseDrawer: $('btn-close-drawer'),
      drawerBackdrop: $('drawer-backdrop'),
      drawerPanel: $('drawer-panel'),
      setSystemPrompt: $('set-system-prompt'),
      rangeTemp: $('range-temp'),
      valTemp: $('val-temp'),
      rangeTopP: $('range-top-p') || $('range-topp'),
      valTopP: $('val-top-p') || $('val-topp'),
      inpMaxTokens: $('inp-max-tokens'),
      btnSaveDrawer: $('btn-save-drawer'),
      // Auth Dialog
      authDialog: $('auth-dialog'),
      btnCloseAuth: $('btn-close-auth'),
      authTitle: $('auth-title'),
      authErr: $('auth-err'),
      authUserInp: $('auth-user-inp'),
      authPassInp: $('auth-pass-inp'),
      btnAuthExec: $('btn-auth-exec'),
      linkToggleMode: $('link-toggle-mode'),
      // Tools & MCP
      btnOpenTools: $('btn-open-tools'),
      navToolsLabel: $('nav-tools-label'),
      btnQuickTools: $('btn-quick-tools'),
      quickToolsLabel: $('quick-tools-label'),
      toolsDialog: $('tools-dialog'),
      btnCloseTools: $('btn-close-tools'),
      tabInbuiltTools: $('tab-inbuilt-tools'),
      tabMcpServers: $('tab-mcp-servers'),
      tabCustomTools: $('tab-custom-tools'),
      paneInbuiltTools: $('pane-inbuilt-tools'),
      paneMcpServers: $('pane-mcp-servers'),
      paneCustomTools: $('pane-custom-tools'),
      inbuiltToolsList: $('inbuilt-tools-list'),
      btnEnableAllTools: $('btn-enable-all-tools'),
      btnDisableAllTools: $('btn-disable-all-tools'),
      mcpSrvName: $('mcp-srv-name'),
      mcpSrvUrl: $('mcp-srv-url'),
      mcpSrvTransport: $('mcp-srv-transport'),
      mcpSrvHeaders: $('mcp-srv-headers'),
      btnAddMcpServer: $('btn-add-mcp-server'),
      mcpServersList: $('mcp-servers-list'),
      customToolName: $('custom-tool-name'),
      customToolHandler: $('custom-tool-handler'),
      customToolDesc: $('custom-tool-desc'),
      customToolSchema: $('custom-tool-schema'),
      btnAddCustomTool: $('btn-add-custom-tool'),
      customToolsList: $('custom-tools-list')
    };

    // Fast API wrapper
    async function apiFetch(path, opts = {}) {
      opts.headers = opts.headers || {};
      if (APP.token) {
        opts.headers['Authorization'] = `Bearer ${APP.token}`;
      } else {
        opts.headers['Authorization'] = 'Bearer guest';
      }
      if (opts.body && typeof opts.body === 'object') {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(opts.body);
      }
      const res = await fetch(path, opts);
      if (res.status === 401 && APP.token && !path.startsWith('/api/auth/')) {
        logout();
        openAuthModal('login');
      }
      return res;
    }

    // Markdown Parser (Tables, Code blocks, Formatting)
    function escapeHtml(t) {
      const d = document.createElement('div');
      d.textContent = t;
      return d.innerHTML;
    }

    function parseMarkdown(md) {
      if (!md) return '';
      const codeBlocks = [];
      let out = md.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const id = 'BLOCK_' + codeBlocks.length;
        codeBlocks.push({ lang: lang || 'code', code: escapeHtml(code.trimEnd()) });
        return id;
      });

      // Handle unclosed code block during active streaming
      out = out.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*)$/g, (match, lang, code) => {
        const id = 'BLOCK_' + codeBlocks.length;
        codeBlocks.push({ lang: lang || 'code', code: escapeHtml(code.trimEnd()) });
        return id;
      });

      out = escapeHtml(out);

      codeBlocks.forEach((b, idx) => {
        const rep = `<pre><div class="code-topbar"><span>${b.lang}</span><button class="btn-copy" onclick="copySnippet(this)">Copy</button></div><code>${b.code}</code></pre>`;
        out = out.replace('BLOCK_' + idx, rep);
      });

      out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
      out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');

      const lines = out.split('\n');
      let inPre = false;
      const formatted = lines.map(line => {
        if (line.includes('<pre>')) inPre = true;
        if (line.includes('</pre>')) { inPre = false; return line; }
        if (inPre) return line;
        if (line.startsWith('&gt; ')) return `<blockquote>${line.slice(5)}</blockquote>`;
        if (line.startsWith('- ') || line.startsWith('* ')) return `<li>${line.slice(2)}</li>`;
        return line ? `<p>${line}</p>` : '';
      });

      return formatted.join('');
    }

    window.copySnippet = function(btn) {
      const code = btn.closest('pre').querySelector('code').textContent;
      navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
      });
    };

    // Mode Switching (Chat vs Completion)
    if (UI.tabNavChat) {
      UI.tabNavChat.addEventListener('click', () => {
        APP.mode = 'chat';
        UI.tabNavChat.classList.add('active');
        if (UI.tabNavComp) UI.tabNavComp.classList.remove('active');
        if (UI.chatInterface) UI.chatInterface.style.display = 'flex';
        if (UI.completionInterface) UI.completionInterface.style.display = 'none';
      });
    }

    if (UI.tabNavComp) {
      UI.tabNavComp.addEventListener('click', () => {
        APP.mode = 'comp';
        UI.tabNavComp.classList.add('active');
        if (UI.tabNavChat) UI.tabNavChat.classList.remove('active');
        if (UI.chatInterface) UI.chatInterface.style.display = 'none';
        if (UI.completionInterface) UI.completionInterface.style.display = 'flex';
      });
    }

    // Sidebar Toggle
    if (UI.btnToggleSidebar) {
      UI.btnToggleSidebar.addEventListener('click', () => {
        if (UI.sidebar) {
          UI.sidebar.classList.toggle('open');
          if (window.innerWidth > 768) {
            UI.sidebar.style.display = UI.sidebar.style.display === 'none' ? 'flex' : 'none';
          }
        }
      });
    }

    // Auth Dialog Handling
    let authIsRegister = false;
    function openAuthModal(mode = 'login') {
      authIsRegister = mode === 'register';
      if (UI.authTitle) UI.authTitle.textContent = authIsRegister ? 'Create Account' : 'Sign In';
      if (UI.btnAuthExec) UI.btnAuthExec.textContent = authIsRegister ? 'Create Account' : 'Sign In';
      if (UI.linkToggleMode) {
        UI.linkToggleMode.textContent = authIsRegister
          ? 'Already have an account? Sign in'
          : "Don't have an account? Create one";
      }
      if (UI.authErr) UI.authErr.style.display = 'none';
      if (UI.authUserInp) UI.authUserInp.value = '';
      if (UI.authPassInp) UI.authPassInp.value = '';
      if (UI.authDialog) {
        UI.authDialog.classList.add('show');
        if (UI.authUserInp) UI.authUserInp.focus();
      }
    }
    function closeAuthModal() {
      if (UI.authDialog) UI.authDialog.classList.remove('show');
    }

    if (UI.btnAuthUser) {
      UI.btnAuthUser.addEventListener('click', () => {
        if (APP.user) {
          if (confirm(`Logged in as ${APP.user.username}. Sign out?`)) {
            logout();
          }
        } else {
          openAuthModal('login');
        }
      });
    }
    if (UI.btnCloseAuth) UI.btnCloseAuth.addEventListener('click', closeAuthModal);
    if (UI.linkToggleMode) {
      UI.linkToggleMode.addEventListener('click', (e) => {
        e.preventDefault();
        openAuthModal(authIsRegister ? 'login' : 'register');
      });
    }

    if (UI.btnAuthExec) {
      UI.btnAuthExec.addEventListener('click', async () => {
        const username = UI.authUserInp ? UI.authUserInp.value.trim() : '';
        const password = UI.authPassInp ? UI.authPassInp.value : '';
        if (!username || !password) return;

        UI.btnAuthExec.disabled = true;
        try {
          const ep = authIsRegister ? '/api/auth/register' : '/api/auth/login';
          const res = await fetch(ep, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
          });
          const data = await res.json();
          if (!res.ok || !data.token) {
            if (UI.authErr) {
              UI.authErr.textContent = data.error || 'Authentication error';
              UI.authErr.style.display = 'block';
            }
            return;
          }

          APP.token = data.token;
          localStorage.setItem('gufo_auth_token', data.token);
          APP.user = data.user;
          updateUserBadge();
          closeAuthModal();
          await loadConversations();
        } catch (err) {
          if (UI.authErr) {
            UI.authErr.textContent = err.message;
            UI.authErr.style.display = 'block';
          }
        } finally {
          if (UI.btnAuthExec) UI.btnAuthExec.disabled = false;
        }
      });
    }

    function updateUserBadge() {
      if (APP.user) {
        if (UI.sidebarUserName) UI.sidebarUserName.textContent = APP.user.username;
        if (UI.authBtnLabel) UI.authBtnLabel.textContent = APP.user.username;
      } else {
        if (UI.sidebarUserName) UI.sidebarUserName.textContent = 'Guest';
        if (UI.authBtnLabel) UI.authBtnLabel.textContent = 'Sign In';
      }
    }

    async function logout() {
      try { await apiFetch('/api/auth/logout', { method: 'POST' }); } catch (e) {}
      APP.token = '';
      APP.user = null;
      localStorage.removeItem('gufo_auth_token');
      updateUserBadge();
      loadConversations();
    }

    // Settings Drawer
    if (UI.btnOpenDrawer) {
      UI.btnOpenDrawer.addEventListener('click', () => {
        if (UI.setSystemPrompt) UI.setSystemPrompt.value = APP.settings.system_prompt;
        if (UI.rangeTemp) UI.rangeTemp.value = APP.settings.temperature;
        if (UI.valTemp) UI.valTemp.textContent = Number(APP.settings.temperature).toFixed(2);
        if (UI.rangeTopP) UI.rangeTopP.value = APP.settings.top_p;
        if (UI.valTopP) UI.valTopP.textContent = Number(APP.settings.top_p).toFixed(2);
        if (UI.inpMaxTokens) UI.inpMaxTokens.value = APP.settings.max_tokens;

        if (UI.drawerBackdrop) UI.drawerBackdrop.classList.add('show');
        if (UI.drawerPanel) UI.drawerPanel.classList.add('show');
      });
    }

    function closeDrawer() {
      if (UI.drawerBackdrop) UI.drawerBackdrop.classList.remove('show');
      if (UI.drawerPanel) UI.drawerPanel.classList.remove('show');
    }
    if (UI.btnCloseDrawer) UI.btnCloseDrawer.addEventListener('click', closeDrawer);
    if (UI.drawerBackdrop) UI.drawerBackdrop.addEventListener('click', closeDrawer);

    if (UI.rangeTemp) {
      UI.rangeTemp.addEventListener('input', e => {
        if (UI.valTemp) UI.valTemp.textContent = Number(e.target.value).toFixed(2);
      });
    }
    if (UI.rangeTopP) {
      UI.rangeTopP.addEventListener('input', e => {
        if (UI.valTopP) UI.valTopP.textContent = Number(e.target.value).toFixed(2);
      });
    }

    if (UI.btnSaveDrawer) {
      UI.btnSaveDrawer.addEventListener('click', async () => {
        if (UI.setSystemPrompt) APP.settings.system_prompt = UI.setSystemPrompt.value;
        if (UI.rangeTemp) APP.settings.temperature = parseFloat(UI.rangeTemp.value);
        if (UI.rangeTopP) APP.settings.top_p = parseFloat(UI.rangeTopP.value);
        if (UI.inpMaxTokens) APP.settings.max_tokens = parseInt(UI.inpMaxTokens.value, 10);

        try {
          await apiFetch('/api/settings', { method: 'POST', body: APP.settings });
        } catch (e) {}
        closeDrawer();
      });
    }

    // Tools & MCP Management
    function openToolsModal(tab = 'inbuilt') {
      switchToolsTab(tab);
      if (UI.toolsDialog) UI.toolsDialog.classList.add('show');
      loadTools();
    }
    function closeToolsModal() {
      if (UI.toolsDialog) UI.toolsDialog.classList.remove('show');
    }
    function switchToolsTab(tab) {
      APP.tools.activeTab = tab;
      const tabs = [
        { key: 'inbuilt', btn: UI.tabInbuiltTools, pane: UI.paneInbuiltTools },
        { key: 'mcp', btn: UI.tabMcpServers, pane: UI.paneMcpServers },
        { key: 'custom', btn: UI.tabCustomTools, pane: UI.paneCustomTools }
      ];
      tabs.forEach(t => {
        if (t.btn) t.btn.classList.toggle('active', t.key === tab);
        if (t.pane) t.pane.style.display = (t.key === tab ? 'flex' : 'none');
      });
    }

    if (UI.btnOpenTools) UI.btnOpenTools.addEventListener('click', () => openToolsModal('inbuilt'));
    if (UI.btnQuickTools) UI.btnQuickTools.addEventListener('click', () => openToolsModal('inbuilt'));
    if (UI.btnCloseTools) UI.btnCloseTools.addEventListener('click', closeToolsModal);
    if (UI.toolsDialog) {
      UI.toolsDialog.addEventListener('click', (e) => {
        if (e.target === UI.toolsDialog) closeToolsModal();
      });
    }
    if (UI.tabInbuiltTools) UI.tabInbuiltTools.addEventListener('click', () => switchToolsTab('inbuilt'));
    if (UI.tabMcpServers) UI.tabMcpServers.addEventListener('click', () => switchToolsTab('mcp'));
    if (UI.tabCustomTools) UI.tabCustomTools.addEventListener('click', () => switchToolsTab('custom'));

    async function loadTools() {
      try {
        const res = await apiFetch('/api/tools');
        if (res.ok) {
          const data = await res.json();
          APP.tools.inbuilt = data.inbuilt || [];
          APP.tools.mcpServers = data.mcp_servers || [];
          APP.tools.customTools = data.custom_tools || [];
          renderInbuiltTools();
          renderMcpServers();
          renderCustomTools();
          updateToolsPill();
        }
      } catch (err) {
        console.error('Failed to load tools:', err);
      }
    }

    function updateToolsPill() {
      const activeInbuilt = (APP.tools.inbuilt || []).filter(t => t.enabled).length;
      const activeCustom = (APP.tools.customTools || []).filter(t => t.enabled).length;
      const total = activeInbuilt + activeCustom;
      if (UI.quickToolsLabel) {
        UI.quickToolsLabel.textContent = `Tools: ${total}`;
      }
      if (UI.navToolsLabel) {
        UI.navToolsLabel.textContent = `Tools & MCP (${total})`;
      }
      if (UI.btnQuickTools) {
        UI.btnQuickTools.classList.toggle('active', total > 0);
      }
    }

    function renderInbuiltTools() {
      if (!UI.inbuiltToolsList) return;
      UI.inbuiltToolsList.innerHTML = '';
      if (!APP.tools.inbuilt || APP.tools.inbuilt.length === 0) {
        UI.inbuiltToolsList.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; padding:8px;">No inbuilt tools available.</div>';
        return;
      }
      APP.tools.inbuilt.forEach(t => {
        const item = document.createElement('div');
        item.className = 'tool-card-item';
        const paramNames = (t.parameters && t.parameters.properties) ? Object.keys(t.parameters.properties).join(', ') : 'none';
        item.innerHTML = `
          <div class="tool-card-left">
            <div class="tool-card-title">
              <span class="tool-tag-badge">llama.cpp</span>
              <span>${escapeHtml(t.name)}</span>
            </div>
            <div class="tool-card-desc">${escapeHtml(t.description || '')}</div>
            <div class="tool-card-params">params: (${escapeHtml(paramNames)})</div>
          </div>
          <div class="tool-card-right">
            <label class="switch-toggle" title="Enable or disable tool">
              <input type="checkbox" ${t.enabled ? 'checked' : ''}>
              <span class="slider"></span>
            </label>
          </div>
        `;
        const chk = item.querySelector('input[type="checkbox"]');
        chk.addEventListener('change', (e) => {
          t.enabled = e.target.checked;
          saveInbuiltTools();
          updateToolsPill();
        });
        UI.inbuiltToolsList.appendChild(item);
      });
    }

    async function saveInbuiltTools() {
      const enabled = (APP.tools.inbuilt || []).filter(t => t.enabled).map(t => t.name);
      try {
        await apiFetch('/api/tools/inbuilt', {
          method: 'POST',
          body: { enabled }
        });
      } catch (err) {
        console.error('Failed to save inbuilt tools config:', err);
      }
    }

    if (UI.btnEnableAllTools) {
      UI.btnEnableAllTools.addEventListener('click', () => {
        (APP.tools.inbuilt || []).forEach(t => t.enabled = true);
        renderInbuiltTools();
        saveInbuiltTools();
        updateToolsPill();
      });
    }
    if (UI.btnDisableAllTools) {
      UI.btnDisableAllTools.addEventListener('click', () => {
        (APP.tools.inbuilt || []).forEach(t => t.enabled = false);
        renderInbuiltTools();
        saveInbuiltTools();
        updateToolsPill();
      });
    }

    function renderMcpServers() {
      if (!UI.mcpServersList) return;
      UI.mcpServersList.innerHTML = '';
      if (!APP.tools.mcpServers || APP.tools.mcpServers.length === 0) {
        UI.mcpServersList.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; padding:8px;">No MCP servers configured yet. Add one above.</div>';
        return;
      }
      APP.tools.mcpServers.forEach(s => {
        const item = document.createElement('div');
        item.className = 'tool-card-item';
        item.innerHTML = `
          <div class="tool-card-left">
            <div class="tool-card-title">
              <span class="tool-tag-badge mcp">MCP</span>
              <span>${escapeHtml(s.name)}</span>
              <span style="font-size:0.7rem; color:var(--text-dim); font-family:var(--font-mono);">${escapeHtml(s.transport.toUpperCase())}</span>
            </div>
            <div class="tool-card-desc" style="font-family:var(--font-mono); font-size:0.75rem;">${escapeHtml(s.url)}</div>
            <div class="tool-card-params">${s.tools_count || 0} tools discovered &bull; last synced: ${escapeHtml(s.last_synced || 'never')}</div>
          </div>
          <div class="tool-card-right" style="gap:8px;">
            <button class="btn-top btn-sync-mcp" style="font-size:0.75rem; padding:3px 7px;" title="Sync / discover tools">Sync</button>
            <label class="switch-toggle" title="Enable or disable server">
              <input type="checkbox" ${s.enabled ? 'checked' : ''}>
              <span class="slider"></span>
            </label>
            <button class="btn-mini btn-del-mcp" style="color:var(--danger); font-size:1.1rem; padding:0 4px;" title="Delete server">&times;</button>
          </div>
        `;
        const syncBtn = item.querySelector('.btn-sync-mcp');
        syncBtn.addEventListener('click', async () => {
          syncBtn.textContent = 'Syncing...';
          try {
            await apiFetch(`/api/mcp/servers/${s.id}/sync`, { method: 'POST' });
            await loadTools();
          } catch(e) {
            alert('Failed to sync MCP server: ' + e.message);
          }
        });
        const chk = item.querySelector('input[type="checkbox"]');
        chk.addEventListener('change', async (e) => {
          try {
            await apiFetch(`/api/mcp/servers/${s.id}`, {
              method: 'PATCH',
              body: { enabled: e.target.checked }
            });
            s.enabled = e.target.checked;
            updateToolsPill();
          } catch(err) {
            console.error(err);
          }
        });
        const delBtn = item.querySelector('.btn-del-mcp');
        delBtn.addEventListener('click', async () => {
          if (confirm(`Remove MCP server "${s.name}"?`)) {
            try {
              await apiFetch(`/api/mcp/servers/${s.id}`, { method: 'DELETE' });
              await loadTools();
            } catch(e) {
              console.error(e);
            }
          }
        });
        UI.mcpServersList.appendChild(item);
      });
    }

    if (UI.btnAddMcpServer) {
      UI.btnAddMcpServer.addEventListener('click', async () => {
        const name = UI.mcpSrvName.value.trim();
        const url = UI.mcpSrvUrl.value.trim();
        const transport = UI.mcpSrvTransport.value;
        const headers = UI.mcpSrvHeaders.value.trim();
        if (!name || !url) {
          alert('Please enter a server name and URL');
          return;
        }
        UI.btnAddMcpServer.disabled = true;
        UI.btnAddMcpServer.textContent = 'Adding...';
        try {
          const res = await apiFetch('/api/mcp/servers', {
            method: 'POST',
            body: { name, url, transport, headers }
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Failed to add server');
          }
          const srv = await res.json();
          try { await apiFetch(`/api/mcp/servers/${srv.id}/sync`, { method: 'POST' }); } catch(_) {}
          UI.mcpSrvName.value = '';
          UI.mcpSrvUrl.value = '';
          UI.mcpSrvHeaders.value = '';
          await loadTools();
        } catch(e) {
          alert('Error adding MCP server: ' + e.message);
        } finally {
          UI.btnAddMcpServer.disabled = false;
          UI.btnAddMcpServer.textContent = 'Add & Discover Tools';
        }
      });
    }

    function renderCustomTools() {
      if (!UI.customToolsList) return;
      UI.customToolsList.innerHTML = '';
      if (!APP.tools.customTools || APP.tools.customTools.length === 0) {
        UI.customToolsList.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; padding:8px;">No custom tools created yet. Define one above.</div>';
        return;
      }
      APP.tools.customTools.forEach(c => {
        const item = document.createElement('div');
        item.className = 'tool-card-item';
        item.innerHTML = `
          <div class="tool-card-left">
            <div class="tool-card-title">
              <span class="tool-tag-badge custom">Custom</span>
              <span>${escapeHtml(c.name)}</span>
            </div>
            <div class="tool-card-desc">${escapeHtml(c.description || '(no description)')}</div>
            <div class="tool-card-params">${c.handler_url ? 'handler: ' + escapeHtml(c.handler_url) : 'standalone function'}</div>
          </div>
          <div class="tool-card-right" style="gap:8px;">
            <label class="switch-toggle" title="Enable or disable tool">
              <input type="checkbox" ${c.enabled ? 'checked' : ''}>
              <span class="slider"></span>
            </label>
            <button class="btn-mini btn-del-tool" style="color:var(--danger); font-size:1.1rem; padding:0 4px;" title="Delete custom tool">&times;</button>
          </div>
        `;
        const chk = item.querySelector('input[type="checkbox"]');
        chk.addEventListener('change', async (e) => {
          c.enabled = e.target.checked;
          updateToolsPill();
        });
        const delBtn = item.querySelector('.btn-del-tool');
        delBtn.addEventListener('click', async () => {
          if (confirm(`Delete custom tool "${c.name}"?`)) {
            try {
              await apiFetch(`/api/custom-tools/${c.id}`, { method: 'DELETE' });
              await loadTools();
            } catch(e) {
              console.error(e);
            }
          }
        });
        UI.customToolsList.appendChild(item);
      });
    }

    if (UI.btnAddCustomTool) {
      UI.btnAddCustomTool.addEventListener('click', async () => {
        const name = UI.customToolName.value.trim();
        const desc = UI.customToolDesc.value.trim();
        const handler = UI.customToolHandler.value.trim();
        let schemaStr = UI.customToolSchema.value.trim();
        if (!name) {
          alert('Please enter a tool function name');
          return;
        }
        if (!schemaStr) {
          schemaStr = '{"type":"object","properties":{}}';
        }
        try {
          JSON.parse(schemaStr);
        } catch(e) {
          alert('Invalid JSON Schema format: ' + e.message);
          return;
        }

        UI.btnAddCustomTool.disabled = true;
        try {
          const res = await apiFetch('/api/custom-tools', {
            method: 'POST',
            body: {
              name,
              description: desc,
              schema_json: schemaStr,
              handler_url: handler
            }
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Failed to add custom tool');
          }
          UI.customToolName.value = '';
          UI.customToolDesc.value = '';
          UI.customToolHandler.value = '';
          UI.customToolSchema.value = '';
          await loadTools();
        } catch(e) {
          alert('Error creating custom tool: ' + e.message);
        } finally {
          UI.btnAddCustomTool.disabled = false;
        }
      });
    }

    // Conversations Management
    async function loadConversations() {
      try {
        const res = await apiFetch('/api/conversations');
        if (res.ok) {
          APP.conversations = await res.json();
          renderChatList();
          if (APP.conversations.length > 0) {
            selectChat(APP.conversations[0].id);
          } else {
            newChat();
          }
        }
      } catch (err) {}
    }

    function renderChatList(filter = '') {
      UI.chatList.innerHTML = '';
      const list = filter
        ? APP.conversations.filter(c => c.title.toLowerCase().includes(filter.toLowerCase()))
        : APP.conversations;

      list.forEach(c => {
        const item = document.createElement('div');
        item.className = 'chat-item' + (c.id === APP.activeId ? ' active' : '');
        item.onclick = () => selectChat(c.id);

        const title = document.createElement('span');
        title.className = 'chat-title-text';
        title.textContent = c.title || 'Untitled chat';

        const actions = document.createElement('div');
        actions.className = 'chat-item-actions';

        const btnRen = document.createElement('button');
        btnRen.className = 'btn-mini';
        btnRen.innerHTML = '&#9998;';
        btnRen.title = 'Rename';
        btnRen.onclick = (e) => {
          e.stopPropagation();
          const t = prompt('Rename chat:', c.title);
          if (t && t.trim()) renameChat(c.id, t.trim());
        };

        const btnDel = document.createElement('button');
        btnDel.className = 'btn-mini';
        btnDel.innerHTML = '&times;';
        btnDel.title = 'Delete';
        btnDel.onclick = (e) => {
          e.stopPropagation();
          if (confirm('Delete this conversation?')) deleteChat(c.id);
        };

        actions.appendChild(btnRen);
        actions.appendChild(btnDel);
        item.appendChild(title);
        item.appendChild(actions);
        UI.chatList.appendChild(item);
      });
    }

    if (UI.convSearchInput) {
      UI.convSearchInput.addEventListener('input', e => renderChatList(e.target.value.trim()));
    }

    async function newChat() {
      const modelName = APP.modelId || (UI.modelPill && UI.modelPill.textContent !== 'loading model...' ? UI.modelPill.textContent : 'default');
      try {
        const res = await apiFetch('/api/conversations', {
          method: 'POST',
          body: {
            title: 'New chat',
            model: modelName,
            system_prompt: APP.settings.system_prompt
          }
        });
        if (res.ok) {
          const created = await res.json();
          APP.conversations.unshift(created);
          renderChatList();
          selectChat(created.id);
          return created.id;
        }
      } catch (err) {}

      // Fallback local conversation
      const localId = 'conv_' + Date.now();
      const localConv = {
        id: localId,
        title: 'New chat',
        model: modelName,
        system_prompt: APP.settings.system_prompt,
        messages: []
      };
      APP.conversations.unshift(localConv);
      renderChatList();
      selectChat(localId);
      return localId;
    }
    if (UI.btnNewChat) UI.btnNewChat.addEventListener('click', newChat);

    async function selectChat(id) {
      APP.activeId = id;
      renderChatList();
      try {
        const res = await apiFetch(`/api/conversations/${id}`);
        if (res.ok) {
          APP.activeConv = await res.json();
          renderMessages(APP.activeConv.messages || []);
        }
      } catch (err) {}
    }

    async function renameChat(id, title) {
      try {
        const res = await apiFetch(`/api/conversations/${id}`, {
          method: 'PATCH',
          body: { title }
        });
        if (res.ok) {
          const item = APP.conversations.find(c => c.id === id);
          if (item) item.title = title;
          renderChatList();
        }
      } catch (err) {}
    }

    async function deleteChat(id) {
      try {
        const res = await apiFetch(`/api/conversations/${id}`, { method: 'DELETE' });
        if (res.ok) {
          APP.conversations = APP.conversations.filter(c => c.id !== id);
          if (APP.activeId === id) {
            if (APP.conversations.length > 0) selectChat(APP.conversations[0].id);
            else newChat();
          } else {
            renderChatList();
          }
        }
      } catch (err) {}
    }

    // Render Messages
    function renderMessages(messages) {
      UI.messagesContainer.innerHTML = '';
      if (!messages || messages.length === 0) {
        UI.messagesContainer.innerHTML = `
          <div class="llama-hero">
            <div class="llama-hero-icon">&#x1F989;</div>
            <div class="llama-hero-title">Gufo Chat</div>
            <div class="llama-hero-subtitle">High-performance LLM inference running natively on AMD Strix Halo gfx1151. Multi-device chat history synced to SQLite database.</div>
          </div>
        `;
        return;
      }

      messages.forEach(m => appendMessage(m));
      scrollDown();
    }

    function appendMessage(m) {
      if (m.role === 'tool') {
        const card = document.createElement('div');
        card.className = 'tool-call-card';
        card.innerHTML = `
          <div class="tool-call-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
            <span class="tool-call-name">🔧 Tool Result: <b>${escapeHtml(m.name || 'tool')}</b></span>
            <span class="tool-status-pill tool-status-success">✓ completed</span>
          </div>
          <div class="tool-output-box collapsed">
            <pre style="margin:0; color:var(--text);">${escapeHtml(m.content || '')}</pre>
          </div>
        `;
        UI.messagesContainer.appendChild(card);
        return card;
      }

      const isUser = m.role === 'user';
      const group = document.createElement('div');
      group.className = `msg-group ${isUser ? 'user' : 'assistant'}`;

      const author = document.createElement('div');
      author.className = `msg-author ${isUser ? 'user' : 'assistant'}`;
      author.textContent = isUser ? 'User' : 'Gufo';

      const body = document.createElement('div');
      body.className = 'msg-body';

      // Thinking Accordion
      if (m.reasoning_content) {
        const accordion = document.createElement('div');
        accordion.className = 'thought-accordion';
        accordion.innerHTML = `
          <div class="thought-summary" onclick="toggleThoughtBlock(this)">
            <span>&#9660; Thought Process</span>
          </div>
          <div class="thought-text collapsed">${escapeHtml(m.reasoning_content)}</div>
        `;
        body.appendChild(accordion);
      }

      if (m.tool_calls && Array.isArray(m.tool_calls)) {
        m.tool_calls.forEach(tc => {
          const fn = tc.function || tc;
          const fnCard = document.createElement('div');
          fnCard.className = 'tool-call-card';
          fnCard.innerHTML = `
            <div class="tool-call-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
              <span class="tool-call-name">⚙️ Called: <b>${escapeHtml(fn.name || 'tool')}</b></span>
              <span class="tool-status-pill tool-status-success">tool call</span>
            </div>
            <div class="tool-output-box collapsed">
              <pre style="margin:0; color:var(--text);">${escapeHtml(typeof fn.arguments === 'object' ? JSON.stringify(fn.arguments, null, 2) : (fn.arguments || ''))}</pre>
            </div>
          `;
          body.appendChild(fnCard);
        });
      }

      const content = document.createElement('div');
      content.className = 'md-content';
      content.innerHTML = parseMarkdown(m.content || '');
      body.appendChild(content);

      // Actions
      const actions = document.createElement('div');
      actions.className = 'msg-actions';
      const btnCopy = document.createElement('button');
      btnCopy.className = 'btn-msg-action';
      btnCopy.innerHTML = '&#128203; Copy';
      btnCopy.onclick = () => navigator.clipboard.writeText(m.content || '');
      actions.appendChild(btnCopy);

      group.appendChild(author);
      group.appendChild(body);
      group.appendChild(actions);

      UI.messagesContainer.appendChild(group);
      return body;
    }

    window.toggleThoughtBlock = function(header) {
      const text = header.nextElementSibling;
      text.classList.toggle('collapsed');
      header.querySelector('span').innerHTML = text.classList.contains('collapsed')
        ? '&#9654; Thought Process'
        : '&#9660; Thought Process';
    };

    function scrollDown() {
      UI.messagesContainer.scrollTop = UI.messagesContainer.scrollHeight;
    }

    // Image upload & paste
    if (UI.btnUploadFile) {
      UI.btnUploadFile.addEventListener('click', () => {
        if (UI.fileUploader) UI.fileUploader.click();
      });
    }
    if (UI.fileUploader) {
      UI.fileUploader.addEventListener('change', e => {
        const file = e.target.files[0];
        if (file) handleImageFile(file);
      });
    }

    function handleImageFile(file) {
      const reader = new FileReader();
      reader.onload = e => {
        APP.attachedImage = e.target.result;
        if (UI.thumbImg) UI.thumbImg.src = APP.attachedImage;
        if (UI.imageBar) UI.imageBar.style.display = 'flex';
      };
      reader.readAsDataURL(file);
    }
    if (UI.btnRemoveThumb) {
      UI.btnRemoveThumb.addEventListener('click', () => {
        APP.attachedImage = null;
        if (UI.imageBar) UI.imageBar.style.display = 'none';
        if (UI.fileUploader) UI.fileUploader.value = '';
      });
    }

    // Auto-expand textarea
    if (UI.chatInput) {
      UI.chatInput.addEventListener('input', () => {
        UI.chatInput.style.height = 'auto';
        UI.chatInput.style.height = Math.min(UI.chatInput.scrollHeight, 160) + 'px';
      });
      UI.chatInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });
    }

    if (UI.btnSend) UI.btnSend.addEventListener('click', sendMessage);
    if (UI.btnStop) {
      UI.btnStop.addEventListener('click', () => {
        if (APP.abortController) APP.abortController.abort();
      });
    }

    function getActiveToolsList() {
      const tools = [];
      if (APP.tools.inbuilt) {
        for (const t of APP.tools.inbuilt) {
          if (t.enabled) {
            tools.push({
              type: 'function',
              function: {
                name: t.name,
                description: t.description || '',
                parameters: t.parameters || { type: 'object', properties: {} }
              }
            });
          }
        }
      }
      if (APP.tools.customTools) {
        for (const c of APP.tools.customTools) {
          if (c.enabled) {
            let params = { type: 'object', properties: {} };
            try { if (c.schema_json) params = JSON.parse(c.schema_json); } catch(_) {}
            tools.push({
              type: 'function',
              function: {
                name: c.name,
                description: c.description || '',
                parameters: params
              }
            });
          }
        }
      }
      return tools;
    }

    // Send Message with Live SSE Streaming & Autonomous Tool Execution
    async function sendMessage() {
      const text = UI.chatInput.value.trim();
      if ((!text && !APP.attachedImage) || APP.isGenerating) return;

      if (!APP.activeId) await newChat();

      // Clear hero if present
      const hero = UI.messagesContainer.querySelector('.llama-hero');
      if (hero) hero.remove();

      appendMessage({ role: 'user', content: text });
      scrollDown();

      // Persist user message to DB
      if (APP.activeId) {
        try {
          await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
            method: 'POST',
            body: { role: 'user', content: text }
          });
        } catch (_) {}
      }

      // Rename chat on first message
      if (APP.activeConv && (!APP.activeConv.messages || APP.activeConv.messages.length === 0)) {
        renameChat(APP.activeId, text.slice(0, 28) + (text.length > 28 ? '...' : ''));
      }

      // Reset Input
      UI.chatInput.value = '';
      UI.chatInput.style.height = 'auto';
      const imageToSend = APP.attachedImage;
      UI.btnRemoveThumb.click();

      // Start Stream UI
      APP.isGenerating = true;
      UI.btnSend.style.display = 'none';
      UI.btnStop.style.display = 'flex';
      UI.slotText.textContent = '1/1 slots (processing)';

      const msgs = [];
      if (APP.settings.system_prompt) {
        msgs.push({ role: 'system', content: APP.settings.system_prompt });
      }
      if (APP.activeConv && APP.activeConv.messages) {
        APP.activeConv.messages.forEach(m => {
          msgs.push({ role: m.role, content: m.content });
        });
      }

      if (imageToSend) {
        msgs.push({
          role: 'user',
          content: [
            { type: 'text', text: text },
            { type: 'image_url', image_url: { url: imageToSend } }
          ]
        });
      } else {
        msgs.push({ role: 'user', content: text });
      }

      const activeTools = getActiveToolsList();
      let turns = 0;
      const MAX_TURNS = 5;

      try {
        while (turns < MAX_TURNS) {
          turns++;

          // Setup Assistant DOM
          const group = document.createElement('div');
          group.className = 'msg-group assistant';
          const author = document.createElement('div');
          author.className = 'msg-author assistant';
          author.textContent = 'Gufo';

          const body = document.createElement('div');
          body.className = 'msg-body';

          const accordion = document.createElement('div');
          accordion.className = 'thought-accordion';
          accordion.style.display = 'none';
          accordion.innerHTML = `
            <div class="thought-summary" onclick="toggleThoughtBlock(this)">
              <span>&#9660; Thinking...</span>
            </div>
            <div class="thought-text"></div>
          `;
          body.appendChild(accordion);

          const content = document.createElement('div');
          content.className = 'md-content';
          body.appendChild(content);

          group.appendChild(author);
          group.appendChild(body);
          UI.messagesContainer.appendChild(group);
          scrollDown();

          let assistantText = '';
          let thoughtText = '';
          let startTime = performance.now();
          let genTokens = 0;
          let accumulatedToolCalls = [];

          APP.abortController = new AbortController();

          try {
            const reqModel = APP.modelId || (UI.modelPill.textContent !== 'loading model...' ? UI.modelPill.textContent : 'default');
            const payload = {
              model: reqModel,
              messages: msgs,
              stream: true,
              temperature: Number(APP.settings.temperature) || 0.6,
              top_p: Number(APP.settings.top_p) || 0.95,
              max_tokens: parseInt(APP.settings.max_tokens, 10) || 4096
            };
            if (activeTools.length > 0) {
              payload.tools = activeTools;
              payload.tool_choice = 'auto';
            }

            const res = await fetch('/v1/chat/completions', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                ...(APP.token ? { 'Authorization': `Bearer ${APP.token}` } : {})
              },
              body: JSON.stringify(payload),
              signal: APP.abortController.signal
            });

            if (!res.ok) {
              let errMsg = `HTTP ${res.status}`;
              try {
                const errData = await res.json();
                if (errData && errData.error) {
                  errMsg = typeof errData.error === 'string' ? errData.error : (errData.error.message || errMsg);
                }
              } catch (_) {}
              throw new Error(errMsg);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop();

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data:')) continue;
                const dataStr = trimmed.slice(5).trim();
                if (dataStr === '[DONE]') break;

                try {
                  const chunk = JSON.parse(dataStr);
                  const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
                  if (delta) {
                    if (delta.reasoning_content || delta.thought) {
                      const t = delta.reasoning_content || delta.thought;
                      thoughtText += t;
                      accordion.style.display = 'block';
                      accordion.querySelector('.thought-text').textContent = thoughtText;
                      scrollDown();
                    }

                    if (delta.content) {
                      if (thoughtText && !accordion.querySelector('.thought-text').classList.contains('collapsed')) {
                        accordion.querySelector('.thought-text').classList.add('collapsed');
                        accordion.querySelector('span').innerHTML = '&#9654; Thought Process';
                      }

                      assistantText += delta.content;
                      genTokens++;
                      content.innerHTML = parseMarkdown(assistantText);
                      scrollDown();

                      const elapsed = (performance.now() - startTime) / 1000;
                      if (elapsed > 0.5) {
                        const spd = (genTokens / elapsed).toFixed(1);
                        UI.footerSpeedMetrics.textContent = `eval: ${spd} t/s | ${genTokens} tokens`;
                      }
                    }

                    if (delta.tool_calls && Array.isArray(delta.tool_calls)) {
                      delta.tool_calls.forEach(tc => {
                        const idx = tc.index || 0;
                        if (!accumulatedToolCalls[idx]) {
                          accumulatedToolCalls[idx] = {
                            id: tc.id || ('call_' + Math.random().toString(36).slice(2, 9)),
                            name: (tc.function && tc.function.name) || '',
                            args: (tc.function && tc.function.arguments) || ''
                          };
                        } else {
                          if (tc.function && tc.function.name) accumulatedToolCalls[idx].name += tc.function.name;
                          if (tc.function && tc.function.arguments) accumulatedToolCalls[idx].args += tc.function.arguments;
                        }
                      });
                    }
                  }
                } catch (e) {}
              }
            }
          } catch (err) {
            if (err.name === 'AbortError') {
              content.innerHTML += '<p style="color:var(--text-dim); font-style:italic;">[generation stopped]</p>';
            } else {
              content.innerHTML += `<p style="color:var(--danger);">[Error: ${escapeHtml(err.message)}]</p>`;
            }
            break;
          }

          // Check if model emitted text-based tool calls e.g. <tool_call>{"name":"...","arguments":{...}}</tool_call>
          if (accumulatedToolCalls.length === 0 && assistantText.includes('<tool_call>')) {
            const matches = [...assistantText.matchAll(/<tool_call>([\s\S]*?)<\/tool_call>/g)];
            for (const m of matches) {
              try {
                const parsed = JSON.parse(m[1].trim());
                if (parsed.name) {
                  accumulatedToolCalls.push({
                    id: 'call_' + Math.random().toString(36).slice(2, 9),
                    name: parsed.name,
                    args: typeof parsed.arguments === 'object' ? JSON.stringify(parsed.arguments) : (parsed.arguments || '{}')
                  });
                }
              } catch (_) {}
            }
          }

          // Persist assistant message to DB
          if (assistantText || thoughtText) {
            if (APP.activeId) {
              try {
                await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
                  method: 'POST',
                  body: {
                    role: 'assistant',
                    content: assistantText,
                    reasoning_content: thoughtText
                  }
                });
              } catch (_) {}
            }
          }

          // If no tool calls were requested, we are done with the loop!
          if (accumulatedToolCalls.length === 0) {
            break;
          }

          // Execute tool calls
          msgs.push({
            role: 'assistant',
            content: assistantText,
            tool_calls: accumulatedToolCalls.map(tc => ({
              id: tc.id,
              type: 'function',
              function: { name: tc.name, arguments: tc.args }
            }))
          });

          for (const tc of accumulatedToolCalls) {
            if (!tc.name) continue;
            let parsedArgs = {};
            try {
              parsedArgs = JSON.parse(tc.args || '{}');
            } catch (_) {
              parsedArgs = { raw: tc.args };
            }

            const toolCard = document.createElement('div');
            toolCard.className = 'tool-call-card running';
            toolCard.innerHTML = `
              <div class="tool-call-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
                <div style="display:flex; align-items:center; gap:6px;">
                  <span class="tool-call-spinner">⚙️</span>
                  <span class="tool-call-name">Running <b>${escapeHtml(tc.name)}</b></span>
                </div>
                <span class="tool-status-pill tool-status-running">executing...</span>
              </div>
              <div class="tool-output-box collapsed">
                <div style="color:var(--text-dim); margin-bottom:4px; font-size:0.75rem;">Arguments:</div>
                <pre style="margin:0 0 6px 0; color:var(--text);">${escapeHtml(JSON.stringify(parsedArgs, null, 2))}</pre>
                <div class="tool-result-display" style="color:var(--text-dim); font-size:0.75rem;">Waiting for result...</div>
              </div>
            `;
            UI.messagesContainer.appendChild(toolCard);
            scrollDown();

            let resultStr = '';
            let isSuccess = true;
            try {
              const execRes = await apiFetch('/api/tools/execute', {
                method: 'POST',
                body: {
                  tool_name: tc.name,
                  arguments: parsedArgs,
                  conversation_id: APP.activeId
                }
              });
              const execData = await execRes.json();
              isSuccess = execData.success;
              resultStr = typeof execData.result === 'object' ? JSON.stringify(execData.result, null, 2) : String(execData.result);
            } catch (e) {
              isSuccess = false;
              resultStr = 'Tool execution error: ' + e.message;
            }

            toolCard.classList.remove('running');
            const statusPill = toolCard.querySelector('.tool-status-pill');
            if (statusPill) {
              statusPill.className = `tool-status-pill ${isSuccess ? 'tool-status-success' : 'tool-status-error'}`;
              statusPill.textContent = isSuccess ? '✓ completed' : '✗ failed';
            }
            const resultDisplay = toolCard.querySelector('.tool-result-display');
            if (resultDisplay) {
              resultDisplay.innerHTML = `<div style="color:var(--text-dim); margin-bottom:4px; font-size:0.75rem;">Result:</div><pre style="margin:0; color:${isSuccess ? 'var(--text)' : 'var(--danger)'};">${escapeHtml(resultStr)}</pre>`;
            }

            msgs.push({
              role: 'tool',
              tool_call_id: tc.id,
              name: tc.name,
              content: resultStr
            });

            if (APP.activeId) {
              try {
                await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
                  method: 'POST',
                  body: { role: 'tool', content: `[${tc.name}]: ${resultStr}` }
                });
              } catch (_) {}
            }
          }
        }
      } finally {
        APP.isGenerating = false;
        APP.abortController = null;
        UI.btnSend.style.display = 'flex';
        UI.btnStop.style.display = 'none';
        UI.slotText.textContent = '1/1 slots (idle)';
        if (APP.activeId) {
          try {
            const res = await apiFetch(`/api/conversations/${APP.activeId}`);
            if (res.ok) APP.activeConv = await res.json();
          } catch (_) {}
        }
      }
    }

    // Raw Completion Handling
    if (UI.btnCompClear) {
      UI.btnCompClear.addEventListener('click', () => {
        if (UI.compInput) UI.compInput.value = '';
        if (UI.compStatus) UI.compStatus.textContent = '';
      });
    }

    if (UI.btnCompGenerate) {
      UI.btnCompGenerate.addEventListener('click', async () => {
        const prompt = UI.compInput ? UI.compInput.value : '';
        if (!prompt) return;

        UI.btnCompGenerate.disabled = true;
        if (UI.compStatus) UI.compStatus.textContent = 'generating...';

      try {
        const reqModel = APP.modelId || (UI.modelPill.textContent !== 'loading model...' ? UI.modelPill.textContent : 'default');
        const res = await fetch('/completion', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(APP.token ? { 'Authorization': `Bearer ${APP.token}` } : {})
          },
          body: JSON.stringify({
            prompt: prompt,
            model: reqModel,
            stream: false,
            n_predict: parseInt(APP.settings.max_tokens, 10) || 4096,
            temperature: Number(APP.settings.temperature) || 0.6,
            top_p: Number(APP.settings.top_p) || 0.95
          })
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error?.message || data.error || `HTTP ${res.status}`);
        }
        if (data.content !== undefined) {
          UI.compInput.value += data.content;
          UI.compStatus.textContent = `done: ${data.tokens_predicted || 0} tokens generated`;
        }
      } catch (err) {
        UI.compStatus.textContent = 'Error: ' + err.message;
      } finally {
        UI.btnCompGenerate.disabled = false;
      }
    });
    }

    // Query Models & Slots
    async function fetchModelAndSlots() {
      const headers = APP.token ? { 'Authorization': `Bearer ${APP.token}` } : {};
      try {
        const res = await fetch('/v1/models', { headers });
        if (res.ok) {
          const d = await res.json();
          if (d.data && d.data[0] && d.data[0].id) {
            APP.modelId = d.data[0].id;
            UI.modelPill.textContent = d.data[0].id;
          }
        }
      } catch (e) {}

      if (!APP.modelId) {
        try {
          const resReady = await fetch('/v1/ready', { headers });
          if (resReady.ok) {
            const rd = await resReady.json();
            if (rd.model) {
              APP.modelId = rd.model;
              UI.modelPill.textContent = rd.model;
            }
          }
        } catch (e) {}
      }

      try {
        const resSlots = await fetch('/slots', { headers });
        if (resSlots.ok) {
          const slots = await resSlots.json();
          if (Array.isArray(slots)) {
            UI.slotText.textContent = `${slots.length}/${slots.length} slots (idle)`;
          }
        }
      } catch (e) {}
    }

    // Init Session
    async function init() {
      await fetchModelAndSlots();
      await loadTools();
      if (APP.token) {
        try {
          const res = await apiFetch('/api/auth/me');
          if (res.ok) {
            const data = await res.json();
            APP.user = data.user;
            if (data.settings) APP.settings = { ...APP.settings, ...data.settings };
            updateUserBadge();
          }
        } catch (e) {}
      } else {
        updateUserBadge();
      }
      await loadConversations();
    }

    init();
  </script>
</body>
</html>
)rawhtml";

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

}  // namespace gufo::server

#endif  // GUFO_SERVER_CHAT_PORTAL_HTML_HPP_
