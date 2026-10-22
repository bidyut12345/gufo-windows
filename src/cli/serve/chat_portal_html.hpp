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

    /* Modern sleek minimalist dark scrollbar */
    * {
      scrollbar-width: thin;
      scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
    }
    *::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    *::-webkit-scrollbar-track {
      background: transparent;
    }
    *::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.18);
      border-radius: 9999px;
      transition: background-color 0.2s ease;
    }
    *::-webkit-scrollbar-thumb:hover {
      background: rgba(255, 255, 255, 0.35);
    }
    *::-webkit-scrollbar-corner {
      background: transparent;
    }

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
    aside#sidebar.collapsed {
      display: none !important;
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
    .sidebar-view-selector {
      display: none;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 4px;
      background: var(--bg-card);
      padding: 3px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
    }
    .sidebar-view-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 0.72rem;
      font-weight: 500;
      padding: 6px 4px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      transition: all 0.15s;
    }
    .sidebar-view-btn.active {
      background: var(--bg-elevated);
      color: var(--text);
      font-weight: 600;
    }
    @media (max-width: 768px) {
      .sidebar-view-selector {
        display: grid;
      }
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
      position: relative;
    }

    .btn-scroll-bottom {
      position: absolute;
      bottom: 110px;
      left: 50%;
      transform: translateX(-50%) translateY(8px);
      width: 38px;
      height: 38px;
      border-radius: 50%;
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      color: var(--text-dim);
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
      display: none;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      z-index: 25;
      opacity: 0;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .btn-scroll-bottom:hover {
      background: var(--bg-elevated);
      color: var(--text);
      border-color: var(--accent);
      transform: translateX(-50%) translateY(0);
    }
    .btn-scroll-bottom.visible {
      display: flex;
      opacity: 1;
      transform: translateX(-50%) translateY(0);
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
      border-radius: var(--radius);
      font-size: 0.93rem;
      line-height: 1.6;
      position: relative;
    }
    .msg-group.user .msg-body {
      background: #181c24;
      border: 1px solid #27303f;
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

    /* Tool Execution Cards & Results */
    .tool-call-card {
      margin: 8px 0;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: rgba(24, 24, 27, 0.65);  
      font-size: 0.85rem;
      max-width: 100%;
    }
    .tool-call-card.running {
      border-color: rgba(59, 130, 246, 0.4);
    }
    .tool-call-header {
      padding: 8px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--text-muted);
      font-size: 0.8rem;
      font-family: var(--font-mono);
      cursor: pointer;
      user-select: none;
      background: rgba(39, 39, 42, 0.5);
      border-bottom: 1px solid transparent;
      transition: background 0.15s ease;
    }
    .tool-call-header:hover {
      background: var(--bg-elevated);
      color: var(--text);
    }
    .tool-call-name {
      display: flex;
      align-items: center;
      gap: 6px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .tool-status-pill {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      padding: 2px 7px;
      border-radius: 9999px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .tool-status-pill.tool-status-success {
      background: rgba(34, 197, 94, 0.15);
      color: #4ade80;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .tool-status-pill.tool-status-error {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .tool-status-pill.tool-status-running {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .tool-output-box {
      padding: 10px 14px;
      color: var(--text-dim);
      font-size: 0.84rem;
      line-height: 1.5;
      border-top: 1px solid var(--border);
      background: rgba(9, 9, 11, 0.5);
      max-width: 100%;
      overflow-x: hidden;
    }
    .tool-output-box.collapsed {
      display: none;
    }
    .tool-output-box pre,
    .tool-result-display pre,
    .approval-body pre {
      margin: 0;
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 0.8rem;
      line-height: 1.45;
      white-space: pre-wrap !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
      max-width: 100%;
    }

    /* Markdown inside messages */
    .md-content {
      padding: 12px 16px;
    }
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

    /* Message Actions & Usage Badge */
    .msg-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 6px;
      flex-wrap: wrap;
    }
    .btn-msg-action {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 0.72rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      transition: all 0.15s ease;
    }
    .btn-msg-action:hover {
      background: var(--bg-card);
      color: var(--text-muted);
      border-color: var(--border-strong);
    }
    .btn-msg-action.btn-resume {
      color: #60a5fa;
      border-color: rgba(96, 165, 250, 0.4);
      background: rgba(96, 165, 250, 0.08);
      font-weight: 500;
    }
    .btn-msg-action.btn-resume:hover {
      background: rgba(96, 165, 250, 0.18);
      color: #93c5fd;
      border-color: rgba(96, 165, 250, 0.6);
    }
    .btn-msg-action.btn-resume svg {
      fill: currentColor;
      stroke: none;
    }
    .btn-msg-action svg {
      width: 12px;
      height: 12px;
      stroke-width: 2;
    }
    .btn-msg-action:hover svg {
      stroke: var(--text);
    }
    .msg-edit-box {
      width: 100%;
      margin-top: 6px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .msg-edit-textarea {
      width: 100%;
      min-height: 80px;
      max-height: 400px;
      padding: 10px 12px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 0.93rem;
      line-height: 1.5;
      resize: vertical;
      outline: none;
      box-sizing: border-box;
    }
    .msg-edit-textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    .msg-usage-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.72rem;
      color: var(--text-dim);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 2px 8px;
      font-family: var(--font-mono);
      white-space: nowrap;
    }
    .msg-usage-badge .usage-item {
      display: inline-flex;
      align-items: center;
      gap: 3px;
    }
    .msg-usage-badge .usage-val {
      color: var(--text-muted);
      font-weight: 500;
    }
    .msg-usage-badge .usage-accent {
      color: var(--accent);
      font-weight: 600;
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
      align-items: center;
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      padding: 3px 6px;
      gap: 10px;
      flex-wrap: wrap;
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

    /* Monitor View */
    .monitor-view {
      display: none;
      flex: 1;
      flex-direction: column;
      padding: 20px 24px;
      max-width: 1040px;
      width: 100%;
      margin: 0 auto;
      gap: 16px;
      overflow-y: auto;
    }
    .monitor-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .monitor-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .monitor-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .monitor-live-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.72rem;
      color: var(--success);
      background: rgba(34, 197, 94, 0.1);
      border: 1px solid rgba(34, 197, 94, 0.25);
      border-radius: var(--radius-sm);
      padding: 3px 8px;
      font-family: var(--font-mono);
    }
    .monitor-pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.4; }
      50% { opacity: 1; transform: scale(1.15); }
      100% { opacity: 0.4; }
    }
    .monitor-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }
    .monitor-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .monitor-card:hover {
      border-color: var(--border-strong);
      transform: translateY(-1px);
    }
    .monitor-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.76rem;
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .monitor-card-value {
      font-size: 1.85rem;
      font-weight: 700;
      font-family: var(--font-mono);
      color: var(--text);
      line-height: 1.1;
      display: flex;
      align-items: baseline;
      gap: 6px;
    }
    .monitor-card-unit {
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-muted);
    }
    .monitor-card-footer {
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: 2px;
    }
    .monitor-slots-container {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .monitor-slots-title {
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .slot-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.82rem;
      flex-wrap: wrap;
      gap: 8px;
    }
    .slot-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .slot-badge.idle {
      background: rgba(113, 113, 122, 0.15);
      color: var(--text-muted);
    }
    .slot-badge.busy {
      background: rgba(59, 130, 246, 0.18);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .progress-bar-bg {
      background: var(--bg-elevated);
      border-radius: 4px;
      height: 6px;
      width: 100%;
      overflow: hidden;
      margin-top: 4px;
    }
    .progress-bar-fill {
      background: var(--accent);
      height: 100%;
      transition: width 0.3s ease;
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
      width: 100%;
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

    .sidebar-backdrop {
      display: none !important;
    }
    .btn-close-sidebar-mobile {
      display: none;
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 1.3rem;
      cursor: pointer;
      padding: 4px 8px;
      line-height: 1;
    }

    @media (max-width: 768px) {
      header.top-nav {
        padding: 0 10px;
        height: 52px;
        justify-content: space-between;
      }
      .nav-left {
        gap: 8px;
        min-width: 0;
        flex: 0 0 auto;
      }
      .brand-llama {
        font-size: 0.95rem;
        gap: 6px;
      }
      .brand-badge {
        display: none !important;
      }
      /* Hide top header tabs on mobile to prevent overlap; accessible via sidebar */
      .nav-tabs {
        display: none !important;
      }
      .nav-right {
        gap: 6px;
        min-width: 0;
        flex: 0 0 auto;
      }
      .nav-right .btn-top span {
        display: none !important;
      }
      .nav-right .btn-top {
        padding: 5px 7px;
        min-width: 32px;
        height: 32px;
      }
      #btn-nav-workdir {
        display: none !important;
      }
      #slot-indicator {
        display: none !important;
      }
      #model-pill {
        max-width: 90px;
        padding: 3px 6px;
        font-size: 0.68rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .btn-close-sidebar-mobile {
        display: block;
      }
      .sidebar-backdrop {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.45);
        z-index: 95;
      }
      .sidebar-backdrop.show {
        display: block !important;
      }
      aside#sidebar {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 290px;
        max-width: 85vw;
        z-index: 100;
        transform: translateX(-100%);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: none;
      }
      aside#sidebar.open {
        transform: translateX(0);
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.8);
      }
      .chat-bottom {
        padding: 0 8px 8px;
        width: 100%;
        max-width: 100%;
      }
      .input-box {
        padding: 6px 10px;
        border-radius: 12px;
        width: 100%;
      }
      .input-row {
        gap: 6px;
        align-items: center;
      }
      #quick-tools-container {
        padding: 3px 6px;
        gap: 4px;
        border-radius: 14px;
        flex-shrink: 0;
      }
      #btn-quick-tools-gear {
        display: none !important;
      }
      .tools-label-prefix {
        display: none !important;
      }
      #chat-input {
        font-size: 15px;
        min-height: 24px;
        line-height: 1.4;
      }
      .btn-icon-input {
        width: 30px;
        height: 30px;
      }
      .btn-submit, .btn-stop-stream {
        width: 32px;
        height: 32px;
      }
      .chat-messages {
        padding: 12px 10px;
      }
      .msg-group {
        max-width: 100%;
      }
      .chat-stats-footer {
        flex-direction: column;
        gap: 2px;
        align-items: flex-start;
      }
      .tools-pill {
        padding: 3px 6px;
        font-size: 0.7rem;
      }
      #quick-tools-label {
        font-size: 0.7rem;
      }
      .monitor-view {
        padding: 12px 10px;
        gap: 12px;
      }
      .monitor-grid {
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }
      .monitor-card {
        padding: 12px 10px;
      }
      .monitor-card-value {
        font-size: 1.35rem;
      }
      .completion-view {
        padding: 12px 10px;
      }
      .drawer-panel {
        width: 100vw;
        max-width: 100%;
      }
      .tools-dialog-box {
        width: 95%;
        max-height: 90vh;
      }
    }

    @media (max-width: 480px) {
      .monitor-grid {
        grid-template-columns: 1fr;
      }
      .nav-tabs .tab-item {
        padding: 3px 6px;
        font-size: 0.7rem;
      }
    }
    /* Drawer Tabs */
    .drawer-tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: var(--bg-card);
      padding: 0 16px;
      gap: 8px;
    }
    .drawer-tab-btn {
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--text-muted);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 10px 12px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .drawer-tab-btn:hover {
      color: var(--text);
    }
    .drawer-tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }

    /* Attachment Menu & Tools Popover */
    .attach-menu-container {
      position: relative;
      display: inline-flex;
      align-items: center;
      flex-shrink: 0;
    }
    .attach-tools-badge {
      position: absolute;
      top: 3px;
      right: 3px;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #22c55e;
      border: 1.5px solid var(--bg-card);
      transition: all 0.2s;
      pointer-events: none;
    }
    .attach-tools-badge.disabled {
      background: var(--text-dim);
      opacity: 0.5;
    }
    .attach-popup-menu {
      position: absolute;
      bottom: calc(100% + 10px);
      left: 0;
      width: 320px;
      max-width: calc(100vw - 32px);
      max-height: 82vh;
      overflow-y: auto;
      background: rgba(20, 20, 25, 0.97);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.05);
      padding: 8px;
      z-index: 100;
      display: flex;
      flex-direction: column;
      gap: 3px;
      animation: attachPopIn 0.16s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes attachPopIn {
      from { opacity: 0; transform: translateY(6px) scale(0.97); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .attach-popup-header {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      padding: 6px 10px 4px 10px;
    }
    .attach-popup-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      border-radius: var(--radius);
      cursor: pointer;
      user-select: none;
      transition: all 0.12s ease;
      color: var(--text);
    }
    .attach-popup-item:hover {
      background: var(--bg-hover);
    }
    .attach-item-icon {
      font-size: 1.15rem;
      width: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .attach-item-info {
      flex: 1;
      min-width: 0;
    }
    .attach-item-title {
      font-size: 0.84rem;
      font-weight: 500;
      color: var(--text);
      line-height: 1.2;
    }
    .attach-item-sub {
      font-size: 0.72rem;
      color: var(--text-dim);
      margin-top: 2px;
      line-height: 1.2;
    }
    .attach-path-truncate {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 190px;
      font-family: var(--font-mono);
    }
    .attach-item-arrow {
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-left: auto;
    }
    .attach-popup-divider {
      height: 1px;
      background: var(--border);
      margin: 4px 6px;
    }
    .attach-popup-card {
      padding: 6px 10px;
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.04);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .attach-card-label {
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .btn-mini-arrow {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 0.68rem;
      padding: 2px 6px;
      cursor: pointer;
      border-radius: var(--radius-sm);
      transition: all 0.15s ease;
      line-height: 1;
    }
    .btn-mini-arrow:hover {
      background: var(--bg-hover);
      color: var(--text);
    }
    .attach-subgroup-container {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 6px;
      background: rgba(0, 0, 0, 0.35);
      border-radius: var(--radius);
      border: 1px solid rgba(255, 255, 255, 0.05);
      margin: 2px 2px 4px 2px;
    }
    .attach-group-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 8px;
      border-radius: var(--radius-sm);
      transition: background 0.12s;
    }
    .attach-group-row:hover {
      background: rgba(255, 255, 255, 0.04);
    }
    .attach-group-info {
      display: flex;
      align-items: center;
      gap: 7px;
    }
    .attach-group-title {
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--text);
    }
    .attach-group-count {
      font-size: 0.68rem;
      color: var(--text-dim);
    }
    .popup-select {
      width: 100%;
      background: var(--bg-card);
      color: var(--text);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      padding: 6px 8px;
      font-size: 0.8rem;
      font-family: var(--font-sans);
      outline: none;
      cursor: pointer;
    }
    .popup-select:focus {
      border-color: var(--accent);
    }
    .btn-mini-action {
      background: var(--bg-hover);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.7rem;
      font-weight: 500;
      padding: 2px 7px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-mini-action:hover {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .segmented-control {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      background: var(--bg-elevated);
      border-radius: var(--radius-sm);
      padding: 2px;
      gap: 2px;
      border: 1px solid var(--border);
    }
    @media (max-width: 480px) {
      .segmented-control {
        grid-template-columns: repeat(3, 1fr);
      }
    }
    .seg-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 0.72rem;
      font-weight: 500;
      padding: 5px 2px;
      border-radius: 3px;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    .seg-btn:hover {
      color: var(--text);
    }
    .seg-btn.active {
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }
    .context-meter-bg {
      width: 100%;
      height: 5px;
      background: var(--bg-elevated);
      border-radius: 9999px;
      overflow: hidden;
      margin: 4px 0 2px 0;
    }
    .context-meter-bar {
      height: 100%;
      background: linear-gradient(90deg, #3b82f6, #8b5cf6);
      border-radius: 9999px;
      transition: width 0.3s ease;
    }

    /* Hero Attachment & Context Dashboard Grid */
    .hero-attach-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      width: 100%;
      max-width: 880px;
      margin: 20px auto 0;
      text-align: left;
    }
    .hero-attach-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      transition: all 0.2s ease;
      position: relative;
      cursor: pointer;
    }
    .hero-attach-card:hover {
      border-color: var(--border-strong);
      background: var(--bg-elevated);
    }
    .hero-attach-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .hero-attach-card-icon {
      font-size: 1.25rem;
    }
    .hero-attach-card-title {
      font-size: 0.86rem;
      font-weight: 600;
      color: var(--text);
    }
    .hero-attach-card-sub {
      font-size: 0.72rem;
      color: var(--text-dim);
      line-height: 1.35;
    }
    .hero-card-badge {
      font-size: 0.68rem;
      font-weight: 500;
      padding: 2px 8px;
      border-radius: 9999px;
      background: var(--bg-hover);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    .tag-pill {
      font-size: 0.65rem;
      padding: 1px 6px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }
    .hero-prompt-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 6px;
      margin-top: 6px;
    }
    .hero-prompt-thumb {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 6px 8px;
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
    }
    .hero-prompt-thumb:hover {
      background: var(--bg-hover);
      border-color: var(--border-strong);
      transform: translateY(-1px);
    }
    .hero-prompt-thumb.active {
      background: rgba(124, 58, 237, 0.15);
      border-color: var(--accent);
      box-shadow: 0 0 0 1px var(--accent);
    }
    .hero-prompt-thumb-icon {
      font-size: 1.25rem;
      line-height: 1;
      flex-shrink: 0;
    }
    .hero-prompt-thumb-info {
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-width: 0;
    }
    .hero-prompt-thumb-title {
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .hero-prompt-thumb-sub {
      font-size: 0.65rem;
      color: var(--text-dim);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .chat-meta-indicators {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.69rem;
      color: var(--text-dim);
      font-family: var(--font-sans);
      flex-wrap: wrap;
    }
    .meta-indicator {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      cursor: pointer;
      padding: 1px 4px;
      border-radius: var(--radius-sm);
      transition: all 0.15s ease;
      color: var(--text-dim);
    }
    .meta-indicator:hover {
      color: var(--text);
      background: var(--bg-hover);
    }
    .meta-sep {
      opacity: 0.35;
      font-size: 0.6rem;
    }
    .settings-prompt-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 6px;
    }
    .settings-prompt-item {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .settings-prompt-item:hover {
      border-color: var(--border-strong);
      background: var(--bg-elevated);
    }
    .settings-prompt-item.active {
      border-color: var(--accent);
      background: rgba(124, 58, 237, 0.06);
    }
    .settings-prompt-item-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .settings-prompt-icon {
      font-size: 1.2rem;
      line-height: 1;
      flex-shrink: 0;
    }
    .settings-prompt-title {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .settings-prompt-desc {
      font-size: 0.68rem;
      color: var(--text-dim);
      line-height: 1.2;
    }
    .settings-prompt-actions {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
    }
    .settings-prompt-preview {
      font-size: 0.72rem;
      color: var(--text-muted);
      line-height: 1.4;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: var(--radius-sm);
      padding: 6px 8px;
      max-height: 64px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--font-mono);
    }

    /* Human-in-the-loop Model Inquiry Cards */
    .inquiry-card {
      border-color: rgba(59, 130, 246, 0.4) !important;
      background: rgba(59, 130, 246, 0.05) !important;
    }
    .inquiry-opt-btn {
      background: var(--bg-elevated);
      border: 1px solid var(--border-strong);
      color: var(--text);
      font-size: 0.82rem;
      padding: 6px 14px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
      font-weight: 500;
    }
    .inquiry-opt-btn:hover {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }

    /* File Explorer Sidebar */
    .file-item-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 8px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      font-size: 0.8rem;
      transition: background 0.15s;
    }
    .file-item-row:hover {
      background: var(--bg-hover);
    }
    .btn-compact-context {
      background: rgba(139, 92, 246, 0.18);
      border: 1px solid rgba(139, 92, 246, 0.35);
      color: #c4b5fd;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .btn-compact-context:hover {
      background: rgba(139, 92, 246, 0.35);
      border-color: #8b5cf6;
      color: #fff;
    }
    .attached-files-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 6px 12px 2px 12px;
    }
    .attached-file-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      font-size: 0.75rem;
      color: var(--text);
      animation: attachPopIn 0.15s ease;
    }
    .attached-file-pill .file-size {
      color: var(--text-dim);
      font-size: 0.7rem;
    }
    .attached-file-pill .btn-remove-file {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 0.95rem;
      cursor: pointer;
      line-height: 1;
      padding: 0 2px;
    }
    .attached-file-pill .btn-remove-file:hover {
      color: var(--danger);
    }
    .attach-popup-tools-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 10px;
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.03);
    }

    /* Tools & MCP Styles */
    .switch-toggle-mini {
      position: relative;
      display: inline-block;
      width: 26px;
      height: 14px;
      flex-shrink: 0;
    }
    .switch-toggle-mini input { opacity: 0; width: 0; height: 0; }
    .slider-mini {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: var(--bg-hover);
      border-radius: 14px;
      transition: .2s;
      border: 1px solid var(--border);
    }
    .slider-mini:before {
      position: absolute;
      content: "";
      height: 8px;
      width: 8px;
      left: 2px;
      bottom: 2px;
      background-color: #a1a1aa;
      border-radius: 50%;
      transition: .2s;
    }
    input:checked + .slider-mini {
      background-color: #22c55e;
      border-color: #22c55e;
    }
    input:checked + .slider-mini:before {
      background-color: white;
      transform: translateX(12px);
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

    /* User Profile Dropdown Menu in Top Right */
    .user-menu-wrapper {
      position: relative;
    }
    .user-popup-menu {
      position: absolute;
      top: calc(100% + 8px);
      right: 0;
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.05);
      min-width: 250px;
      padding: 6px;
      z-index: 100;
      display: flex;
      flex-direction: column;
      gap: 3px;
      animation: popoverFadeIn 0.15s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .user-popup-header {
      padding: 8px 12px 10px 12px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 4px;
    }
    .user-menu-item {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      padding: 8px 10px;
      background: transparent;
      border: none;
      border-radius: var(--radius-sm);
      color: var(--text);
      cursor: pointer;
      text-decoration: none;
      transition: background-color 0.12s ease;
      font-size: 0.82rem;
    }
    .user-menu-item:hover {
      background: var(--bg-hover);
      color: #fff;
    }
    .user-menu-icon {
      font-size: 1.15rem;
      width: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .user-menu-text {
      flex: 1;
      text-align: left;
    }
    .user-menu-title {
      font-weight: 500;
      line-height: 1.2;
    }
    .user-menu-desc {
      font-size: 0.7rem;
      color: var(--text-dim);
      margin-top: 2px;
    }
    .user-menu-divider {
      height: 1px;
      background: var(--border);
      margin: 4px 6px;
    }

    /* Modal File Browser Styles */
    .fb-breadcrumbs-bar {
      display: flex;
      align-items: center;
      gap: 4px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
    }
    .fb-crumb {
      color: var(--accent);
      cursor: pointer;
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      transition: background 0.12s;
    }
    .fb-crumb:hover {
      background: rgba(59, 130, 246, 0.15);
      text-decoration: underline;
    }
    .fb-crumb.active {
      color: var(--text-muted);
      cursor: default;
      pointer-events: none;
    }
    .fb-crumb-sep {
      color: var(--text-dim);
      user-select: none;
    }
    .fb-file-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 7px 10px;
      border-radius: var(--radius-sm);
      border: 1px solid transparent;
      background: rgba(255, 255, 255, 0.015);
      transition: all 0.12s ease;
    }
    .fb-file-row:hover {
      background: var(--bg-hover);
      border-color: var(--border);
    }
    .fb-file-left {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
      min-width: 0;
      cursor: pointer;
    }
    .fb-file-name {
      font-size: 0.82rem;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .fb-file-name.is-folder {
      font-weight: 600;
      color: #93c5fd;
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
        <button id="tab-nav-monitor" class="tab-item">Monitor</button>
      </div>
    </div>

    <div class="nav-right">
      <div id="slot-indicator" class="slot-pill" title="Server Slots Status">
        <span class="slot-dot"></span>
        <span id="slot-text">1/1 slots (idle)</span>
      </div>
      <div id="model-pill" class="model-pill" title="Loaded Model">loading model...</div>
      <button id="btn-open-drawer" class="btn-top" title="Parameters & Settings">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        <span>Settings</span>
      </button>
      <div class="user-menu-wrapper">
        <button id="btn-auth-user" class="btn-top" title="User Account & Workspace Options">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
          <span id="auth-btn-label">Sign In</span>
          <span style="font-size:0.6rem; color:var(--text-dim); margin-left:2px;">&#9660;</span>
        </button>
        <div id="user-popup-menu" class="user-popup-menu" style="display:none;">
          <div class="user-popup-header">
            <div id="user-popup-name" style="font-weight:600; font-size:0.86rem; color:var(--text);">Guest</div>
            <div id="user-popup-root" style="font-size:0.72rem; color:var(--text-dim); font-family:var(--font-mono);">UserData/Guest</div>
          </div>
          <button id="btn-menu-file-browser" class="user-menu-item" type="button">
            <span class="user-menu-icon">📁</span>
            <div class="user-menu-text">
              <div class="user-menu-title">Workspace File Browser</div>
              <div class="user-menu-desc">Explore, upload &amp; manage chat files</div>
            </div>
          </button>
          <div class="user-menu-divider"></div>
          <button id="btn-menu-auth" class="user-menu-item" type="button">
            <span id="user-menu-auth-icon" class="user-menu-icon">🔑</span>
            <div class="user-menu-text">
              <div id="user-menu-auth-title" class="user-menu-title">Sign In / Switch User</div>
              <div id="user-menu-auth-desc" class="user-menu-desc">Multi-device sync &amp; isolation</div>
            </div>
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Workspace -->
  <div class="main-workspace">
    <!-- Left Sidebar -->
    <aside id="sidebar">
      <div class="sidebar-top">
        <div style="display:flex; align-items:center; gap:8px; width:100%;">
          <button id="btn-new-chat" class="btn-new-chat" style="flex:1;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
            <span>New chat</span>
          </button>
          <button id="btn-close-sidebar-mobile" class="btn-close-sidebar-mobile" title="Close sidebar">&times;</button>
        </div>
        <div class="sidebar-view-selector">
          <button id="side-nav-chat" class="sidebar-view-btn active" type="button">
            <span>💬</span> Chat
          </button>
          <button id="side-nav-comp" class="sidebar-view-btn" type="button">
            <span>📝</span> Completion
          </button>
          <button id="side-nav-monitor" class="sidebar-view-btn" type="button">
            <span>📊</span> Monitor
          </button>
        </div>
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
    <div id="sidebar-backdrop" class="sidebar-backdrop"></div>

    <!-- Main Viewport -->
    <main class="content-area">
      <!-- Chat Interface -->
      <div id="chat-interface" class="chat-view">
        <div id="messages-container" class="chat-messages">
          <!-- Rendered dynamically -->
        </div>

        <button id="btn-scroll-bottom" class="btn-scroll-bottom" title="Scroll to bottom" type="button">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
        </button>

        <div class="chat-bottom">
          <div class="input-box">
            <div id="image-bar" class="image-preview-bar">
              <img id="thumb-img" class="image-thumb" src="" alt="preview">
              <button id="btn-remove-thumb" class="btn-remove-thumb">&times;</button>
            </div>
            <div id="attached-files-bar" class="attached-files-bar" style="display:none;"></div>
            <div class="input-row">
              <input type="file" id="file-uploader-image" accept="image/*" style="display:none">
              <input type="file" id="file-uploader-doc" accept=".txt,.md,.json,.py,.cpp,.h,.hpp,.c,.js,.ts,.html,.css,.log,.csv,.pdf,.yaml,.toml,.rs,.go,.sh,.bat,.cmd,.xml,.ini,.cfg" multiple style="display:none">
              
              <!-- Attachment Button & Popover Menu -->
              <div class="attach-menu-container">
                <button id="btn-upload-file" class="btn-icon-input" title="Context, Attachments & Tools" type="button">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
                  <span id="attach-tools-badge" class="attach-tools-badge" title="Agent Tools Status"></span>
                </button>

                <!-- Attachment & Context Popover Menu (All 7 Menu Items) -->
                <div id="attach-popup-menu" class="attach-popup-menu" style="display:none;">
                  <div class="attach-popup-header">Attachments &amp; Context</div>
                  
                  <!-- 1. Attach images (to be added in context) -->
                  <div id="attach-item-image" class="attach-popup-item">
                    <span class="attach-item-icon">🖼️</span>
                    <div class="attach-item-info">
                      <div class="attach-item-title">Attach Images</div>
                      <div class="attach-item-sub">Add visual context for multimodal vision</div>
                    </div>
                  </div>

                  <!-- 2. Attach Files -->
                  <div id="attach-item-files" class="attach-popup-item">
                    <span class="attach-item-icon">📄</span>
                    <div class="attach-item-info">
                      <div class="attach-item-title">Attach Files</div>
                      <div class="attach-item-sub">Code, markdown, json, logs, or text files</div>
                    </div>
                  </div>

                  <!-- 3. Working directory (Show selected directory name from server) -->
                  <div id="attach-item-workdir" class="attach-popup-item">
                    <span class="attach-item-icon">📁</span>
                    <div class="attach-item-info">
                      <div class="attach-item-title">Working Directory</div>
                      <div id="attach-popup-workdir-sub" class="attach-item-sub attach-path-truncate">UserData/...</div>
                    </div>
                    <span class="attach-item-arrow">⚙️</span>
                  </div>

                  <div class="attach-popup-divider"></div>

                  <!-- 4. Enable / disable all mcp tools -->
                  <div class="attach-popup-tools-row">
                    <div style="display:flex; align-items:center; gap:8px;">
                      <span style="font-size:1.05rem;">⚡</span>
                      <div>
                        <div style="font-size:0.83rem; font-weight:600; color:var(--text);">Agent &amp; MCP Tools</div>
                        <div id="attach-tools-status-text" style="font-size:0.7rem; color:var(--text-dim);">8 tools enabled</div>
                      </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                      <button id="btn-toggle-tools-submenu" class="btn-mini-arrow" title="Toggle tool groups submenu" type="button">Groups ▼</button>
                      <label class="switch-toggle-mini" title="Toggle all tools ON / OFF" style="cursor:pointer; margin:0;" onclick="event.stopPropagation();">
                        <input type="checkbox" id="chk-master-tools" checked>
                        <span class="slider-mini"></span>
                      </label>
                    </div>
                  </div>

                  <!-- 4.a - Submenu (list all mcp tool groups, can enable and disable) -->
                  <div id="attach-tools-groups-sub" class="attach-subgroup-container" style="display:none;">
                    <div style="font-size:0.68rem; font-weight:600; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.04em; padding:2px 4px;">Tool Groups</div>
                    <div id="attach-tool-groups-list" style="display:flex; flex-direction:column; gap:2px;"></div>
                    <div style="display:flex; justify-content:flex-end; padding-top:2px;">
                      <button id="attach-btn-manage-tools" class="btn-mini-action" type="button" style="font-size:0.68rem;">Advanced MCP Settings ➔</button>
                    </div>
                  </div>

                  <div class="attach-popup-divider"></div>

                  <!-- 5. System prompt drop down and a button to add new -->
                  <div class="attach-popup-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <div class="attach-card-label">System Prompt</div>
                      <button id="btn-popup-new-prompt" class="btn-mini-action" type="button">+ New</button>
                    </div>
                    <select id="popup-system-prompt-select" class="popup-select">
                      <option value="default">Gufo Assistant (Default)</option>
                      <option value="coder">Expert Software Engineer</option>
                      <option value="concise">Concise &amp; Direct</option>
                      <option value="reasoner">Deep Reasoning &amp; Logic</option>
                      <option value="creative">Creative &amp; Brainstorm</option>
                    </select>
                  </div>

                  <div class="attach-popup-divider"></div>

                  <!-- Tool Permission: Ask / Allow all -->
                  <div class="attach-popup-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <div class="attach-card-label">Tool Permission</div>
                      <span id="popup-perm-label" style="font-size:0.68rem; color:var(--text-dim);">Ask before run</span>
                    </div>
                    <div class="segmented-control" id="tool-perm-control" style="grid-template-columns:1fr 1fr; margin-top:4px;" onclick="event.stopPropagation()">
                      <button type="button" class="seg-btn active" data-perm="ask" onclick="setToolPermission('ask'); event.stopPropagation();" title="Prompt for approval before running tools">🛡️ Ask</button>
                      <button type="button" class="seg-btn" data-perm="allow" onclick="setToolPermission('allow'); event.stopPropagation();" title="Allow autonomous tool execution without asking">⚡ Allow all</button>
                    </div>
                  </div>

                  <div class="attach-popup-divider"></div>

                  <!-- 6. Thinking level (auto, minimum, low, medium, high, xhigh) -->
                  <div class="attach-popup-card">
                    <div class="attach-card-label">Thinking Level</div>
                    <div class="segmented-control" id="thinking-level-control" onclick="event.stopPropagation()">
                      <button type="button" class="seg-btn active" data-level="auto" onclick="setThinkingLevel('auto'); event.stopPropagation();" title="Default thinking budget">Auto</button>
                      <button type="button" class="seg-btn" data-level="minimum" onclick="setThinkingLevel('minimum'); event.stopPropagation();" title="Fast/concise reasoning (512 tokens)">Min</button>
                      <button type="button" class="seg-btn" data-level="low" onclick="setThinkingLevel('low'); event.stopPropagation();" title="Low reasoning budget (2,048 tokens)">Low</button>
                      <button type="button" class="seg-btn" data-level="medium" onclick="setThinkingLevel('medium'); event.stopPropagation();" title="Medium reasoning budget (4,096 tokens)">Med</button>
                      <button type="button" class="seg-btn" data-level="high" onclick="setThinkingLevel('high'); event.stopPropagation();" title="High reasoning budget (8,192 tokens)">High</button>
                      <button type="button" class="seg-btn" data-level="xhigh" onclick="setThinkingLevel('xhigh'); event.stopPropagation();" title="Maximum reasoning budget (16,384 tokens)">XHigh</button>
                    </div>
                  </div>

                  <div class="attach-popup-divider"></div>

                  <!-- 7. Current context size and compact button -->
                  <div class="attach-popup-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <div class="attach-card-label">Current Context</div>
                      <span id="popup-context-tokens" style="font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono);">0 / 262k</span>
                    </div>
                    <div class="context-meter-bg">
                      <div id="popup-context-bar" class="context-meter-bar" style="width:0%;"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                      <span id="popup-context-pct" style="font-size:0.68rem; color:var(--text-dim);">0.0% used</span>
                      <button id="btn-compact-context" class="btn-compact-context" type="button" title="Compact earlier conversation turns into concise summary">🧹 Compact</button>
                    </div>
                  </div>
                </div>
              </div>

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
            <div class="chat-meta-indicators">
              <span id="input-meta-prompt" class="meta-indicator" onclick="cycleSystemPrompt()" title="Click to cycle prompt persona">🦉 Gufo</span>
              <span class="meta-sep">•</span>
              <span id="input-meta-thinking" class="meta-indicator" onclick="cycleThinkingLevel()" title="Click to cycle thinking level">🧠 Auto</span>
              <span class="meta-sep">•</span>
              <span id="input-meta-perm" class="meta-indicator" onclick="cycleToolPermission()" title="Click to toggle tool permission">🛡️ Ask</span>
              <span class="meta-sep">•</span>
              <span id="input-meta-ctx" class="meta-indicator" onclick="openAttachPopup()" title="Click to view context meter">📊 0 ctx</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span id="footer-speed-metrics">eval: 0.0 t/s | prompt: 0.0 t/s</span>
            </div>
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

      <!-- Monitor Interface -->
      <div id="monitor-interface" class="monitor-view">
        <div class="monitor-header">
          <div class="monitor-title">
            <span>📊</span>
            <span>Model Telemetry &amp; System Monitor</span>
          </div>
          <div class="monitor-actions">
            <span class="monitor-live-tag">
              <span class="monitor-pulse-dot"></span>
              <span id="monitor-live-status">Live Telemetry</span>
            </span>
            <button id="btn-refresh-monitor" class="btn-top" title="Refresh metrics" style="padding:4px 10px;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
              <span>Refresh</span>
            </button>
          </div>
        </div>

        <!-- 6 Main KPI Stat Cards -->
        <div class="monitor-grid">
          <!-- Card 1: Active Messages Handled -->
          <div class="monitor-card">
            <div class="monitor-card-header">
              <span>Active Messages Handled</span>
              <span>⚡</span>
            </div>
            <div class="monitor-card-value">
              <span id="mon-val-active-msg">0</span>
              <span class="monitor-card-unit">in-flight</span>
            </div>
            <div class="monitor-card-footer">
              <span id="mon-sub-queue">Queue: 0 deferred</span> &bull; <span id="mon-sub-slots">1/1 slots idle</span>
            </div>
          </div>

          <!-- Card 2: Generation Speed -->
          <div class="monitor-card">
            <div class="monitor-card-header">
              <span>Token Generation Speed</span>
              <span>🚀</span>
            </div>
            <div class="monitor-card-value">
              <span id="mon-val-gen-spd">0.0</span>
              <span class="monitor-card-unit">t/s</span>
            </div>
            <div class="monitor-card-footer">
              <span id="mon-sub-gen-type">Current / Last decode speed</span>
            </div>
          </div>

          <!-- Card 3: Prompt Speed -->
          <div class="monitor-card">
            <div class="monitor-card-header">
              <span>Prompt Processing Speed</span>
              <span>🏎️</span>
            </div>
            <div class="monitor-card-value">
              <span id="mon-val-prompt-spd">0.0</span>
              <span class="monitor-card-unit">prompt t/s</span>
            </div>
            <div class="monitor-card-footer">
              <span id="mon-sub-prompt-type">Prefill &amp; prompt throughput</span>
            </div>
          </div>

          <!-- Card 4: Generated Tokens Count -->
          <div class="monitor-card">
            <div class="monitor-card-header">
              <span>Tokens Generated</span>
              <span>✍️</span>
            </div>
            <div class="monitor-card-value">
              <span id="mon-val-gen-tokens">0</span>
              <span class="monitor-card-unit">tokens</span>
            </div>
            <div class="monitor-card-footer">
              <span>Total session: </span><b id="mon-sub-total-gen">0</b>
            </div>
          </div>

          <!-- Card 5: Prompt Tokens Count -->
          <div class="monitor-card">
            <div class="monitor-card-header">
              <span>Prompt Tokens Count</span>
              <span>📥</span>
            </div>
            <div class="monitor-card-value">
              <span id="mon-val-prompt-tokens">0</span>
              <span class="monitor-card-unit">prompt</span>
            </div>
            <div class="monitor-card-footer">
              <span>Total: </span><b id="mon-sub-total-prompt">0</b> &bull; <span style="color:var(--success);">Cached: </span><b id="mon-sub-cached-prompt" style="color:var(--success);">0</b>
            </div>
          </div>

          <!-- Card 6: Speculative Decoding & Hardware -->
          <div class="monitor-card">
            <div class="monitor-card-header">
              <span>Speculative Decoding &amp; RAM</span>
              <span>🧠</span>
            </div>
            <div class="monitor-card-value">
              <span id="mon-val-spec-acc">0%</span>
              <span class="monitor-card-unit">draft acc</span>
            </div>
            <div class="monitor-card-footer">
              <span id="mon-sub-drafts">Drafts: 0/0</span> &bull; <span id="mon-sub-mem">gfx1151</span>
            </div>
          </div>
        </div>

        <!-- Real-Time Slot Activity Card -->
        <div class="monitor-slots-container">
          <div class="monitor-slots-title">
            <span>Hardware Execution Slots</span>
            <span id="mon-slots-summary" style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">1 slot allocated</span>
          </div>
          <div id="mon-slots-list" style="display:flex; flex-direction:column; gap:8px;">
            <!-- Rendered dynamically -->
          </div>
        </div>
      </div>
    </main>
  </div>

  <!-- Settings Drawer (right side dialog) -->
  <div id="drawer-backdrop" class="drawer-backdrop"></div>
  <aside id="drawer-panel" class="drawer-panel">
    <div class="drawer-header">
      <span class="drawer-title">Server &amp; Portal Settings</span>
      <button id="btn-close-drawer" class="btn-top" style="padding:2px 8px;">&times;</button>
    </div>

    <!-- Drawer Navigation Tabs -->
    <div class="drawer-tabs">
      <button id="drawer-tab-inference" class="drawer-tab-btn active" type="button">Sampling &amp; Model</button>
      <button id="drawer-tab-tools" class="drawer-tab-btn" type="button">⚡ Tools &amp; MCP</button>
    </div>

    <!-- Tab 1: Inference & Sampling -->
    <div id="drawer-pane-inference" class="drawer-body">
      <div class="setting-group">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <div>
            <label class="setting-label" style="margin:0;">System Prompt &amp; Personas</label>
            <span class="setting-desc" style="display:block; margin-top:2px;">Manage AI personas, custom behavior prompts, and default context.</span>
          </div>
          <button id="btn-settings-add-prompt" class="btn-dialog-submit" type="button" style="padding:4px 10px; font-size:0.75rem; white-space:nowrap;">+ Add New</button>
        </div>

        <!-- Currently Active Prompt Box -->
        <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:10px; margin-bottom:12px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="font-size:0.72rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em;">Active System Prompt</div>
            <span id="settings-active-prompt-tag" class="hero-card-badge">🦉 Gufo Assistant</span>
          </div>
          <textarea id="set-system-prompt" class="setting-textarea" rows="3" placeholder="System prompt instructions..."></textarea>
          <div style="display:flex; justify-content:flex-end; gap:6px; margin-top:6px;">
            <button id="btn-save-active-prompt-text" class="btn-top" type="button" style="padding:3px 10px; font-size:0.72rem; color:var(--accent);">Apply Text</button>
          </div>
        </div>

        <!-- List of System Prompts -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:0.76rem; font-weight:600; color:var(--text);">Prompt Library &amp; Presets</span>
          <span id="settings-prompts-count" style="font-size:0.7rem; color:var(--text-dim); font-family:var(--font-mono);"></span>
        </div>
        <div id="settings-prompt-list" class="settings-prompt-list"></div>
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
        <div style="font-weight:600; font-size:0.85rem; color:var(--text); margin-bottom:4px;">Server &amp; Storage</div>
        <div style="font-size:0.75rem; color:var(--text-dim); line-height:1.4;">
          Database: <code style="color:var(--text-muted);">%LOCALAPPDATA%\gufo\chat_portal.db</code><br>
          Acceleration: <span style="color:#86efac;">AMD Strix Halo gfx1151</span>
        </div>
      </div>

      <button id="btn-save-drawer" class="btn-dialog-submit" style="margin-top:10px;">Save Settings</button>
    </div>

    <!-- Tab 2: Tools & MCP Settings -->
    <div id="drawer-pane-tools" class="drawer-body" style="display:none;">
      <!-- Tools Subtabs -->
      <div class="tools-tabs" style="background:transparent; padding:0; border-bottom:1px solid var(--border); margin-top:0;">
        <button id="tab-inbuilt-tools" class="tools-tab-btn active" type="button">Inbuilt Tools</button>
        <button id="tab-mcp-servers" class="tools-tab-btn" type="button">MCP Servers</button>
        <button id="tab-custom-tools" class="tools-tab-btn" type="button">Custom Tools</button>
      </div>

      <!-- Inbuilt Tools Pane -->
      <div id="pane-inbuilt-tools" class="tools-content-pane" style="padding:4px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:0.78rem; color:var(--text-muted);">Standard built-in system tools executed on host server.</span>
          <div style="display:flex; gap:6px;">
            <button id="btn-enable-all-tools" class="btn-top" type="button" style="font-size:0.72rem; padding:2px 6px;">Enable All</button>
            <button id="btn-disable-all-tools" class="btn-top" type="button" style="font-size:0.72rem; padding:2px 6px;">Disable All</button>
          </div>
        </div>
        <div id="inbuilt-tools-list" style="display:flex; flex-direction:column; gap:8px;"></div>
      </div>

      <!-- MCP Servers Pane -->
      <div id="pane-mcp-servers" class="tools-content-pane" style="display:none; padding:4px 0;">
        <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.4;">
          Connect external Model Context Protocol (MCP) servers via Server-Sent Events (SSE) or HTTP stream. Tools provided by these servers are automatically exposed to Gufo Chat.
        </div>
        
        <div style="background:var(--bg-elevated); padding:10px; border-radius:var(--radius); display:flex; flex-direction:column; gap:8px;">
          <div style="font-weight:600; font-size:0.82rem;">Add MCP Server</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
            <input type="text" id="mcp-srv-name" class="setting-input" placeholder="Server Name (e.g. Memory)">
            <input type="text" id="mcp-srv-url" class="setting-input" placeholder="URL (e.g. http://localhost:8001/sse)">
          </div>
          <div style="display:grid; grid-template-columns:100px 1fr; gap:6px;">
            <select id="mcp-srv-transport" class="setting-input">
              <option value="sse">SSE</option>
              <option value="http">HTTP</option>
            </select>
            <input type="text" id="mcp-srv-headers" class="setting-input" placeholder='Headers JSON {"Auth":"..."}'>
          </div>
          <button id="btn-add-mcp-server" class="btn-dialog-submit" type="button" style="padding:6px; font-size:0.8rem;">Add &amp; Discover Tools</button>
        </div>

        <div style="font-weight:600; font-size:0.82rem; margin-top:4px;">Configured MCP Servers</div>
        <div id="mcp-servers-list" style="display:flex; flex-direction:column; gap:6px;"></div>
      </div>

      <!-- Custom Tools Pane -->
      <div id="pane-custom-tools" class="tools-content-pane" style="display:none; padding:4px 0;">
        <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.4;">
          Define custom function tools with JSON Schema parameters. When called by the model, Gufo can invoke an HTTP handler URL or return arguments.
        </div>

        <div style="background:var(--bg-elevated); padding:10px; border-radius:var(--radius); display:flex; flex-direction:column; gap:8px;">
          <div style="font-weight:600; font-size:0.82rem;">Add Custom Function Tool</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
            <input type="text" id="custom-tool-name" class="setting-input" placeholder="Name (e.g. query_db)">
            <input type="text" id="custom-tool-handler" class="setting-input" placeholder="Handler URL (optional)">
          </div>
          <input type="text" id="custom-tool-desc" class="setting-input" placeholder="Description of tool...">
          <textarea id="custom-tool-schema" class="setting-input" style="height:60px; font-family:var(--font-mono); font-size:0.72rem;" placeholder='{"type":"object","properties":{...}}'></textarea>
          <button id="btn-add-custom-tool" class="btn-dialog-submit" type="button" style="padding:6px; font-size:0.8rem;">Add Custom Tool</button>
        </div>

        <div style="font-weight:600; font-size:0.82rem; margin-top:4px;">Your Custom Tools</div>
        <div id="custom-tools-list" style="display:flex; flex-direction:column; gap:6px;"></div>
      </div>
    </div>
  </aside>

  <!-- Conversation Working Directory Modal (Dedicated Dialog) -->
  <div id="conv-workdir-dialog" class="dialog-backdrop">
    <div class="dialog-box" style="max-width:540px;">
      <div class="dialog-header">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:1.15rem;">📁</span>
          <span style="font-weight:700; font-size:1rem;">Conversation Working Directory</span>
        </div>
        <button id="btn-close-conv-workdir" class="btn-top" style="padding:2px 8px;">&times;</button>
      </div>
      <div class="dialog-body">
        <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
          Each conversation has its own dedicated workspace on the server. Filesystem tools (read, write, edit, grep, shell) are scoped strictly to this folder.
        </div>

        <div class="setting-group" style="background:var(--bg-elevated); padding:8px 10px; border-radius:var(--radius);">
          <div style="font-size:0.72rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.04em;">Active Conversation</div>
          <div id="conv-workdir-active-title" style="font-weight:600; font-size:0.88rem; color:var(--text); margin-top:2px;">New Chat</div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px; flex-wrap:wrap; gap:4px;">
            <div id="conv-workdir-active-id" style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-dim);"></div>
            <div id="conv-workdir-display-badge" style="font-family:var(--font-mono); font-size:0.75rem; color:#86efac; background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.25); border-radius:4px; padding:2px 6px;">UserData/Guest/</div>
          </div>
        </div>

        <!-- 1. Select from all folders under username -->
        <div class="setting-group">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <label class="setting-label">Select Existing Folder</label>
            <span id="conv-workdir-user-prefix" style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-dim);">UserData/Guest/</span>
          </div>
          <select id="conv-workdir-select" class="popup-select" style="width:100%; font-family:var(--font-mono); font-size:0.82rem;">
            <option value="">-- Choose existing folder under username --</option>
          </select>
          <div style="font-size:0.72rem; color:var(--text-dim); margin-top:2px;">
            Showing all folders created under your sandboxed user directory.
          </div>
        </div>

        <!-- 2. Or enter custom folder name -->
        <div class="setting-group">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <label class="setting-label">Folder Name</label>
            <button id="btn-conv-workdir-reset" class="btn-top" type="button" style="font-size:0.72rem; padding:2px 8px;">Reset to Default</button>
          </div>
          <div style="display:flex; align-items:center; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius-sm); overflow:hidden;">
            <span id="conv-workdir-inp-prefix" style="padding:6px 8px; font-family:var(--font-mono); font-size:0.78rem; color:var(--text-dim); background:var(--bg-elevated); border-right:1px solid var(--border); white-space:nowrap; user-select:none;">UserData/Guest/</span>
            <input type="text" id="conv-workdir-inp" class="setting-input" style="border:none; border-radius:0; font-family:var(--font-mono); font-size:0.82rem; padding:6px 8px; flex:1;" placeholder="<chat folder>">
          </div>
          <div style="font-size:0.72rem; color:var(--text-dim); margin-top:4px; line-height:1.3;">
            Default chat folder: <code id="conv-workdir-default-hint" style="color:var(--text-muted);">conv_xxx - Title</code>
          </div>
        </div>

        <div style="display:flex; gap:8px; margin-top:10px;">
          <button id="btn-conv-workdir-open-browser" class="btn-top" type="button" style="padding:6px 12px; gap:6px;">
            <span>📁</span> Open File Browser
          </button>
          <button id="btn-conv-workdir-save" class="btn-dialog-submit" style="flex:1;">Save &amp; Apply</button>
          <button id="btn-conv-workdir-cancel" class="btn-top" style="padding:0 14px;">Cancel</button>
        </div>
      </div>
    </div>
  </div>

  <!-- New System Prompt Dialog -->
  <div id="new-prompt-dialog" class="dialog-backdrop">
    <div class="dialog-box" style="max-width:500px;">
      <div class="dialog-header">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:1.15rem;">✨</span>
          <span id="new-prompt-dialog-title" style="font-weight:700; font-size:1rem;">Add Custom System Prompt</span>
        </div>
        <button id="btn-close-new-prompt" class="btn-top" style="padding:2px 8px;">&times;</button>
      </div>
      <div class="dialog-body">
        <input type="hidden" id="edit-prompt-idx" value="">
        <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
          Create or edit a personalized system prompt persona. Custom prompts are saved locally and available across conversations.
        </div>
        <div class="setting-group">
          <label class="setting-label">Prompt Name</label>
          <input type="text" id="new-prompt-name-inp" class="setting-input" placeholder="e.g. C++ Kernel Expert">
        </div>
        <div class="setting-group">
          <label class="setting-label">Instructions / Persona</label>
          <textarea id="new-prompt-text-inp" class="setting-textarea" rows="4" placeholder="You are an expert in..."></textarea>
        </div>
        <div style="display:flex; gap:8px; margin-top:8px;">
          <button id="btn-save-new-prompt" class="btn-dialog-submit" style="flex:1;">Save Prompt</button>
          <button id="btn-cancel-new-prompt" class="btn-top" style="padding:0 14px;">Cancel</button>
        </div>
      </div>
    </div>
  </div>

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

  <!-- File Viewer Modal (View/Download sandboxed user files) -->
  <div id="file-viewer-dialog" class="dialog-backdrop">
    <div class="dialog-box" style="max-width:680px; width:90vw;">
      <div class="dialog-header">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>📄</span>
          <span id="file-viewer-title" style="font-weight:700; font-size:0.95rem;">File Preview</span>
        </div>
        <button id="btn-close-file-viewer" class="btn-top" style="padding:2px 8px;">&times;</button>
      </div>
      <div class="dialog-body">
        <pre id="file-viewer-content" style="max-height:55vh; overflow-y:auto; font-family:var(--font-mono); font-size:0.75rem; background:var(--bg); padding:10px; border-radius:var(--radius-sm); border:1px solid var(--border); white-space:pre-wrap; word-break:break-all;"></pre>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
          <span id="file-viewer-meta" style="font-size:0.72rem; color:var(--text-dim);"></span>
          <button id="btn-file-viewer-download" class="btn-dialog-submit" style="padding:4px 14px; font-size:0.8rem;">Download</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Fully Featured Workspace File Browser Modal Dialog -->
  <div id="user-file-browser-dialog" class="dialog-backdrop">
    <div class="dialog-box" style="max-width:880px; width:94vw; max-height:90vh; display:flex; flex-direction:column; padding:0; overflow:hidden;">
      <!-- Modal Header -->
      <div class="dialog-header" style="padding:12px 18px; border-bottom:1px solid var(--border); background:var(--bg-card); display:flex; justify-content:space-between; align-items:center;">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:1.3rem;">📁</span>
          <div>
            <div style="font-weight:700; font-size:1rem; color:var(--text); display:flex; align-items:center; gap:8px;">
              <span>Workspace File Browser</span>
              <span id="fb-user-badge" class="tag-pill" style="font-size:0.7rem;">Guest</span>
            </div>
            <div id="fb-active-chat-workdir" style="font-size:0.72rem; color:var(--text-dim); font-family:var(--font-mono); margin-top:2px;">
              Chat Workdir: <span id="fb-chat-workdir-text" style="color:#86efac;">UserData/Guest/conv_...</span>
            </div>
          </div>
        </div>
        <button id="btn-close-file-browser" class="btn-top" style="padding:4px 10px; font-size:1.1rem; line-height:1;">&times;</button>
      </div>

      <!-- Toolbar & Breadcrumbs -->
      <div style="padding:10px 18px; background:var(--bg-elevated); border-bottom:1px solid var(--border); display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
          <!-- Breadcrumb navigation -->
          <div id="fb-breadcrumbs" class="fb-breadcrumbs-bar" style="display:flex; align-items:center; gap:4px; font-family:var(--font-mono); font-size:0.8rem; overflow-x:auto; flex:1;">
            <span class="fb-crumb active" data-path="">UserData</span>
          </div>

          <!-- Action buttons -->
          <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
            <button id="btn-fb-up" class="btn-top" type="button" title="Go to Parent Directory" style="padding:4px 9px; font-size:0.75rem;">
              <span>⬆️</span> Up
            </button>
            <button id="btn-fb-new-folder" class="btn-top" type="button" title="Create New Subfolder" style="padding:4px 9px; font-size:0.75rem;">
              <span>➕</span> Folder
            </button>
            <button id="btn-fb-upload" class="btn-top" type="button" title="Upload Files to Current Directory" style="padding:4px 9px; font-size:0.75rem;">
              <span>📤</span> Upload
            </button>
            <input type="file" id="fb-file-upload-input" multiple style="display:none;">
            <button id="btn-fb-set-workdir" class="btn-top" type="button" title="Set this folder as Active Chat Working Directory" style="padding:4px 10px; font-size:0.75rem; background:rgba(34,197,94,0.15); border-color:rgba(34,197,94,0.35); color:#86efac; font-weight:600;">
              <span>📌</span> Set as Chat Workdir
            </button>
            <button id="btn-fb-refresh" class="btn-top" type="button" title="Refresh Files" style="padding:4px 8px; font-size:0.75rem;">
              <span>🔄</span>
            </button>
          </div>
        </div>

        <!-- Inline New Folder Form (toggleable) -->
        <div id="fb-new-folder-box" style="display:none; align-items:center; gap:8px; padding:6px 8px; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius-sm);">
          <span style="font-size:0.8rem; color:var(--text-dim);">New Folder Name:</span>
          <input type="text" id="fb-new-folder-inp" class="setting-input" style="padding:3px 8px; font-size:0.8rem; flex:1;" placeholder="e.g. project_code">
          <button id="btn-fb-create-folder-exec" class="btn-dialog-submit" style="padding:3px 10px; font-size:0.75rem;">Create</button>
          <button id="btn-fb-cancel-folder" class="btn-top" style="padding:3px 8px; font-size:0.75rem;">Cancel</button>
        </div>
      </div>

      <!-- Main Browser Area: Files List + Inline Preview -->
      <div style="display:flex; flex:1; min-height:360px; max-height:55vh; overflow:hidden;">
        <!-- Left: Files / Folders List -->
        <div id="fb-files-list-view" style="flex:1; overflow-y:auto; padding:10px 18px; display:flex; flex-direction:column; gap:4px;">
          <!-- Dynamically populated rows -->
        </div>

        <!-- Right: Inline File Preview Pane (collapsible) -->
        <div id="fb-preview-pane" style="display:none; width:380px; border-left:1px solid var(--border); background:var(--bg); flex-direction:column; overflow:hidden;">
          <div style="padding:8px 12px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; background:var(--bg-elevated);">
            <div style="display:flex; align-items:center; gap:6px; min-width:0; overflow:hidden;">
              <span id="fb-preview-icon">📄</span>
              <span id="fb-preview-name" style="font-weight:600; font-size:0.82rem; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"></span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button id="btn-fb-preview-download" class="btn-mini" title="Download File" style="font-size:0.8rem; padding:2px 6px;">⬇️</button>
              <button id="btn-fb-preview-close" class="btn-mini" title="Close Preview" style="font-size:1.1rem; line-height:1; padding:0 4px;">&times;</button>
            </div>
          </div>
          <div id="fb-preview-body" style="flex:1; overflow:auto; padding:10px;">
            <pre id="fb-preview-text" style="margin:0; font-family:var(--font-mono); font-size:0.75rem; color:var(--text); white-space:pre-wrap; word-break:break-all;"></pre>
            <img id="fb-preview-img" style="display:none; max-width:100%; height:auto; border-radius:4px; margin:0 auto;" />
          </div>
          <div id="fb-preview-footer" style="padding:6px 12px; border-top:1px solid var(--border); font-size:0.7rem; color:var(--text-dim); display:flex; justify-content:space-between; align-items:center;">
            <span id="fb-preview-size"></span>
            <button id="btn-fb-preview-copy" class="btn-mini" style="font-size:0.7rem; padding:1px 6px;">📋 Copy</button>
          </div>
        </div>
      </div>

      <!-- Modal Footer -->
      <div style="padding:10px 18px; border-top:1px solid var(--border); background:var(--bg-card); display:flex; justify-content:space-between; align-items:center;">
        <div id="fb-status-stats" style="font-size:0.75rem; color:var(--text-dim);">
          0 items
        </div>
        <div style="display:flex; gap:8px;">
          <button id="btn-fb-close-footer" class="btn-top" style="padding:4px 14px;">Close</button>
        </div>
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
      attachedFiles: [],
      toolPermission: localStorage.getItem('gufo_tool_permission') || 'ask',
      thinkingLevel: localStorage.getItem('gufo_thinking_level') || 'auto',
      customPrompts: JSON.parse(localStorage.getItem('gufo_custom_prompts') || '[]'),
      isGenerating: false,
      abortController: null,
      mode: 'chat', // 'chat' or 'comp'
      toolsEnabled: localStorage.getItem('gufo_tools_enabled') !== 'false',
      serverCurrentDirectory: '.',
      settings: {
        system_prompt: 'You are a helpful, respectful and honest assistant.',
        temperature: 0.6,
        top_p: 0.95,
        max_tokens: 4096,
        working_directory: ''
      },
      tools: {
        inbuilt: [],
        mcpServers: [],
        customTools: [],
        activeTab: 'inbuilt'
      }
    };
    if (!['auto', 'minimum', 'low', 'medium', 'high', 'xhigh'].includes(APP.thinkingLevel)) {
      APP.thinkingLevel = 'auto';
    }

    // DOM references
    const $ = id => document.getElementById(id);
    const UI = {
      tabNavChat: $('tab-nav-chat'),
      tabNavComp: $('tab-nav-comp'),
      tabNavMonitor: $('tab-nav-monitor'),
      sideNavChat: $('side-nav-chat'),
      sideNavComp: $('side-nav-comp'),
      sideNavMonitor: $('side-nav-monitor'),
      filesCwdLabel: $('files-cwd-label'),
      btnFilesUp: $('btn-files-up'),
      btnFilesRefresh: $('btn-files-refresh'),
      filesListContainer: $('files-list-container'),
      fileViewerDialog: $('file-viewer-dialog'),
      fileViewerTitle: $('file-viewer-title'),
      fileViewerContent: $('file-viewer-content'),
      fileViewerMeta: $('file-viewer-meta'),
      btnFileViewerDownload: $('btn-file-viewer-download'),
      btnCloseFileViewer: $('btn-close-file-viewer'),
      toolPermControl: $('tool-perm-control'),
      popupPermLabel: $('popup-perm-label'),
      chatInterface: $('chat-interface'),
      completionInterface: $('completion-interface'),
      monitorInterface: $('monitor-interface'),
      slotText: $('slot-text'),
      modelPill: $('model-pill'),
      btnNavWorkdir: $('btn-nav-workdir'),
      navWorkdirLabel: $('nav-workdir-label'),
      btnToggleSidebar: $('btn-toggle-sidebar'),
      sidebar: $('sidebar'),
      sidebarBackdrop: $('sidebar-backdrop'),
      btnCloseSidebarMobile: $('btn-close-sidebar-mobile'),
      btnRefreshMonitor: $('btn-refresh-monitor'),
      monValActiveMsg: $('mon-val-active-msg'),
      monSubQueue: $('mon-sub-queue'),
      monSubSlots: $('mon-sub-slots'),
      monValGenSpd: $('mon-val-gen-spd'),
      monSubGenType: $('mon-sub-gen-type'),
      monValPromptSpd: $('mon-val-prompt-spd'),
      monSubPromptType: $('mon-sub-prompt-type'),
      monValGenTokens: $('mon-val-gen-tokens'),
      monSubTotalGen: $('mon-sub-total-gen'),
      monValPromptTokens: $('mon-val-prompt-tokens'),
      monSubTotalPrompt: $('mon-sub-total-prompt'),
      monSubCachedPrompt: $('mon-sub-cached-prompt'),
      monValSpecAcc: $('mon-val-spec-acc'),
      monSubDrafts: $('mon-sub-drafts'),
      monSubMem: $('mon-sub-mem'),
      monSlotsList: $('mon-slots-list'),
      monSlotsSummary: $('mon-slots-summary'),
      btnNewChat: $('btn-new-chat'),
      convSearchInput: $('conv-search-input'),
      chatList: $('chat-list'),
      sidebarUserName: $('sidebar-user-name'),
      authBtnLabel: $('auth-btn-label'),
      btnAuthUser: $('btn-auth-user'),
      messagesContainer: $('messages-container'),
      btnScrollBottom: $('btn-scroll-bottom'),
      imageBar: $('image-bar'),
      thumbImg: $('thumb-img'),
      btnRemoveThumb: $('btn-remove-thumb'),
      attachedFilesBar: $('attached-files-bar'),
      fileUploaderImage: $('file-uploader-image'),
      fileUploaderDoc: $('file-uploader-doc'),
      // Attachment & Popover Menu (7 Menu Items)
      btnUploadFile: $('btn-upload-file'),
      attachToolsBadge: $('attach-tools-badge'),
      attachPopupMenu: $('attach-popup-menu'),
      attachItemImage: $('attach-item-image'),
      attachItemFiles: $('attach-item-files'),
      attachItemWorkdir: $('attach-item-workdir'),
      attachPopupWorkdirSub: $('attach-popup-workdir-sub'),
      chkMasterTools: $('chk-master-tools'),
      attachToolsStatusText: $('attach-tools-status-text'),
      btnToggleToolsSubmenu: $('btn-toggle-tools-submenu'),
      attachToolsGroupsSub: $('attach-tools-groups-sub'),
      attachToolGroupsList: $('attach-tool-groups-list'),
      attachBtnManageTools: $('attach-btn-manage-tools'),
      popupSystemPromptSelect: $('popup-system-prompt-select'),
      btnPopupNewPrompt: $('btn-popup-new-prompt'),
      thinkingLevelControl: $('thinking-level-control'),
      popupContextTokens: $('popup-context-tokens'),
      popupContextBar: $('popup-context-bar'),
      popupContextPct: $('popup-context-pct'),
      btnCompactContext: $('btn-compact-context'),
      // New System Prompt Dialog
      newPromptDialog: $('new-prompt-dialog'),
      newPromptDialogTitle: $('new-prompt-dialog-title'),
      btnCloseNewPrompt: $('btn-close-new-prompt'),
      btnCancelNewPrompt: $('btn-cancel-new-prompt'),
      btnSaveNewPrompt: $('btn-save-new-prompt'),
      newPromptNameInp: $('new-prompt-name-inp'),
      newPromptTextInp: $('new-prompt-text-inp'),
      editPromptIdx: $('edit-prompt-idx'),
      btnTopFileBrowser: $('btn-top-file-browser'),
      userPopupMenu: $('user-popup-menu'),
      userPopupName: $('user-popup-name'),
      userPopupRoot: $('user-popup-root'),
      btnMenuFileBrowser: $('btn-menu-file-browser'),
      btnMenuWorkdirSetup: $('btn-menu-workdir-setup'),
      btnMenuAuth: $('btn-menu-auth'),
      userMenuAuthIcon: $('user-menu-auth-icon'),
      userMenuAuthTitle: $('user-menu-auth-title'),
      userMenuAuthDesc: $('user-menu-auth-desc'),
      // File Browser Modal
      userFileBrowserDialog: $('user-file-browser-dialog'),
      btnCloseFileBrowser: $('btn-close-file-browser'),
      btnFbCloseFooter: $('btn-fb-close-footer'),
      fbUserBadge: $('fb-user-badge'),
      fbActiveChatWorkdir: $('fb-active-chat-workdir'),
      fbChatWorkdirText: $('fb-chat-workdir-text'),
      fbBreadcrumbs: $('fb-breadcrumbs'),
      btnFbUp: $('btn-fb-up'),
      btnFbNewFolder: $('btn-fb-new-folder'),
      btnFbUpload: $('btn-fb-upload'),
      fbFileUploadInput: $('fb-file-upload-input'),
      btnFbSetWorkdir: $('btn-fb-set-workdir'),
      btnFbRefresh: $('btn-fb-refresh'),
      fbNewFolderBox: $('fb-new-folder-box'),
      fbNewFolderInp: $('fb-new-folder-inp'),
      btnFbCreateFolderExec: $('btn-fb-create-folder-exec'),
      btnFbCancelFolder: $('btn-fb-cancel-folder'),
      fbFilesListView: $('fb-files-list-view'),
      fbPreviewPane: $('fb-preview-pane'),
      fbPreviewIcon: $('fb-preview-icon'),
      fbPreviewName: $('fb-preview-name'),
      btnFbPreviewDownload: $('btn-fb-preview-download'),
      btnFbPreviewClose: $('btn-fb-preview-close'),
      fbPreviewText: $('fb-preview-text'),
      fbPreviewImg: $('fb-preview-img'),
      fbPreviewSize: $('fb-preview-size'),
      btnFbPreviewCopy: $('btn-fb-preview-copy'),
      fbStatusStats: $('fb-status-stats'),
      // Conversation Working Directory Modal
      convWorkdirDialog: $('conv-workdir-dialog'),
      btnCloseConvWorkdir: $('btn-close-conv-workdir'),
      btnConvWorkdirCancel: $('btn-conv-workdir-cancel'),
      convWorkdirActiveTitle: $('conv-workdir-active-title'),
      convWorkdirActiveId: $('conv-workdir-active-id'),
      convWorkdirDisplayBadge: $('conv-workdir-display-badge'),
      convWorkdirSelect: $('conv-workdir-select'),
      convWorkdirUserPrefix: $('conv-workdir-user-prefix'),
      convWorkdirInpPrefix: $('conv-workdir-inp-prefix'),
      convWorkdirInp: $('conv-workdir-inp'),
      convWorkdirDefaultHint: $('conv-workdir-default-hint'),
      btnConvWorkdirReset: $('btn-conv-workdir-reset'),
      btnConvWorkdirOpenBrowser: $('btn-conv-workdir-open-browser'),
      btnConvWorkdirSave: $('btn-conv-workdir-save'),
      chatInput: $('chat-input'),
      btnSend: $('btn-send'),
      btnStop: $('btn-stop'),
      footerSpeedMetrics: $('footer-speed-metrics'),
      // Completion
      compInput: $('comp-input'),
      btnCompGenerate: $('btn-comp-generate'),
      btnCompClear: $('btn-comp-clear'),
      compStatus: $('comp-status'),
      // Drawer & Tabs
      btnOpenDrawer: $('btn-open-drawer'),
      btnCloseDrawer: $('btn-close-drawer'),
      drawerBackdrop: $('drawer-backdrop'),
      drawerPanel: $('drawer-panel'),
      drawerTabInference: $('drawer-tab-inference'),
      drawerTabTools: $('drawer-tab-tools'),
      drawerPaneInference: $('drawer-pane-inference'),
      drawerPaneTools: $('drawer-pane-tools'),
      setSystemPrompt: $('set-system-prompt'),
      btnSettingsAddPrompt: $('btn-settings-add-prompt'),
      settingsActivePromptTag: $('settings-active-prompt-tag'),
      btnSaveActivePromptText: $('btn-save-active-prompt-text'),
      settingsPromptsCount: $('settings-prompts-count'),
      settingsPromptList: $('settings-prompt-list'),
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
      // Tools & MCP in Drawer
      btnOpenTools: $('btn-open-tools'),
      navToolsLabel: $('nav-tools-label'),
      modalChkMasterTools: $('modal-chk-master-tools'),
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

    // Mode Switching (Chat vs Completion vs Monitor)
    function switchTab(mode) {
      APP.mode = mode;
      if (UI.tabNavChat) UI.tabNavChat.classList.toggle('active', mode === 'chat');
      if (UI.tabNavComp) UI.tabNavComp.classList.toggle('active', mode === 'comp');
      if (UI.tabNavMonitor) UI.tabNavMonitor.classList.toggle('active', mode === 'monitor');
      if (UI.sideNavChat) UI.sideNavChat.classList.toggle('active', mode === 'chat');
      if (UI.sideNavComp) UI.sideNavComp.classList.toggle('active', mode === 'comp');
      if (UI.sideNavMonitor) UI.sideNavMonitor.classList.toggle('active', mode === 'monitor');

      if (UI.chatInterface) UI.chatInterface.style.display = mode === 'chat' ? 'flex' : 'none';
      if (UI.completionInterface) UI.completionInterface.style.display = mode === 'comp' ? 'flex' : 'none';
      if (UI.monitorInterface) UI.monitorInterface.style.display = mode === 'monitor' ? 'flex' : 'none';

      if (mode === 'chat') {
        if (UI.chatList) UI.chatList.style.display = 'flex';
      }

      if (mode === 'monitor') {
        refreshMonitorStats();
        if (!APP.monitorInterval) {
          APP.monitorInterval = setInterval(refreshMonitorStats, 2000);
        }
      } else {
        if (APP.monitorInterval) {
          clearInterval(APP.monitorInterval);
          APP.monitorInterval = null;
        }
      }
    }

    if (UI.tabNavChat) UI.tabNavChat.addEventListener('click', () => switchTab('chat'));
    if (UI.tabNavComp) UI.tabNavComp.addEventListener('click', () => switchTab('comp'));
    if (UI.tabNavMonitor) UI.tabNavMonitor.addEventListener('click', () => switchTab('monitor'));
    if (UI.sideNavChat) UI.sideNavChat.addEventListener('click', () => {
      if (UI.chatList) UI.chatList.style.display = 'flex';
      switchTab('chat');
      toggleSidebar(false);
    });
    if (UI.sideNavComp) UI.sideNavComp.addEventListener('click', () => { switchTab('comp'); toggleSidebar(false); });
    if (UI.sideNavMonitor) UI.sideNavMonitor.addEventListener('click', () => { switchTab('monitor'); toggleSidebar(false); });
    if (UI.btnRefreshMonitor) UI.btnRefreshMonitor.addEventListener('click', () => refreshMonitorStats());

    // User Sandboxed File Explorer Handlers
    let currentFilesPath = '';



    async function loadUserFiles(subpath = '') {
      currentFilesPath = subpath;
      if (!UI.filesListContainer) return;
      UI.filesListContainer.innerHTML = '<div style="color:var(--text-dim); font-size:0.75rem; padding:8px;">Loading files...</div>';
      if (UI.filesCwdLabel) {
        UI.filesCwdLabel.textContent = 'UserData/' + (subpath ? subpath : '');
        UI.filesCwdLabel.title = 'UserData/' + (subpath ? subpath : '');
      }

      try {
        const url = `/api/user-files?path=${encodeURIComponent(subpath)}`;
        const res = await apiFetch(url);
        if (!res.ok) {
          throw new Error(`Failed to load directory (HTTP ${res.status})`);
        }
        const data = await res.json();
        const entries = data.entries || [];
        if (entries.length === 0) {
          UI.filesListContainer.innerHTML = '<div style="color:var(--text-dim); font-size:0.75rem; padding:12px; text-align:center;">Empty directory</div>';
          return;
        }

        UI.filesListContainer.innerHTML = '';
        entries.forEach(entry => {
          const row = document.createElement('div');
          row.className = 'file-item-row';
          const icon = entry.is_directory ? '📁' : (entry.name.match(/\.(png|jpg|jpeg|webp|gif)$/i) ? '🖼️' : '📄');
          const sizeStr = entry.is_directory ? '' : formatBytes(entry.size || 0);

          row.innerHTML = `
            <div class="file-tree-left" style="display:flex; align-items:center; gap:6px; flex:1; overflow:hidden; cursor:pointer;">
              <span style="font-size:1rem;">${icon}</span>
              <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:0.8rem; color:var(--text); font-weight:${entry.is_directory ? '600' : '400'};">${escapeHtml(entry.name)}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              ${sizeStr ? `<span style="font-size:0.7rem; color:var(--text-dim); font-family:var(--font-mono);">${sizeStr}</span>` : ''}
              ${!entry.is_directory ? `<button class="btn-mini btn-file-preview" title="View / Preview" style="padding:1px 5px; font-size:0.72rem;">👁️</button>` : ''}
              <button class="btn-mini btn-file-del" title="Delete" style="padding:1px 5px; font-size:0.72rem; color:var(--danger);">&times;</button>
            </div>
          `;

          const left = row.querySelector('.file-tree-left');
          left.addEventListener('click', () => {
            if (entry.is_directory) {
              const nextPath = subpath ? `${subpath}/${entry.name}` : entry.name;
              loadUserFiles(nextPath);
            } else {
              const relPath = subpath ? `${subpath}/${entry.name}` : entry.name;
              openFilePreviewModal(relPath, entry.name);
            }
          });

          const previewBtn = row.querySelector('.btn-file-preview');
          if (previewBtn) {
            previewBtn.addEventListener('click', (e) => {
              e.stopPropagation();
              const relPath = subpath ? `${subpath}/${entry.name}` : entry.name;
              openFilePreviewModal(relPath, entry.name);
            });
          }

          const delBtn = row.querySelector('.btn-file-del');
          if (delBtn) {
            delBtn.addEventListener('click', async (e) => {
              e.stopPropagation();
              const relPath = subpath ? `${subpath}/${entry.name}` : entry.name;
              if (confirm(`Delete "${entry.name}"?`)) {
                try {
                  const delRes = await apiFetch(`/api/user-files?path=${encodeURIComponent(relPath)}`, { method: 'DELETE' });
                  if (delRes.ok) {
                    loadUserFiles(subpath);
                  } else {
                    const d = await delRes.json();
                    alert('Failed to delete: ' + (d.error || 'Server error'));
                  }
                } catch (err) {
                  alert('Delete failed: ' + err.message);
                }
              }
            });
          }

          UI.filesListContainer.appendChild(row);
        });
      } catch (err) {
        UI.filesListContainer.innerHTML = `<div style="color:var(--danger); font-size:0.75rem; padding:8px;">${escapeHtml(err.message)}</div>`;
      }
    }

    if (UI.btnFilesUp) {
      UI.btnFilesUp.addEventListener('click', () => {
        if (!currentFilesPath) return;
        const parts = currentFilesPath.split(/[\\/]/).filter(Boolean);
        parts.pop();
        loadUserFiles(parts.join('/'));
      });
    }

    if (UI.btnFilesRefresh) {
      UI.btnFilesRefresh.addEventListener('click', () => loadUserFiles(currentFilesPath));
    }

    function formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    async function openFilePreviewModal(relPath, fileName) {
      if (!UI.fileViewerDialog) return;
      if (UI.fileViewerTitle) UI.fileViewerTitle.textContent = fileName || relPath;
      if (UI.fileViewerMeta) UI.fileViewerMeta.textContent = 'Path: UserData/' + relPath;
      if (UI.fileViewerContent) UI.fileViewerContent.textContent = 'Loading file content...';
      if (UI.btnFileViewerDownload) {
        UI.btnFileViewerDownload.href = `/api/user-files/content?path=${encodeURIComponent(relPath)}`;
        UI.btnFileViewerDownload.download = fileName;
      }
      UI.fileViewerDialog.classList.add('show');

      try {
        const res = await apiFetch(`/api/user-files/content?path=${encodeURIComponent(relPath)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const cType = res.headers.get('content-type') || '';
        if (cType.startsWith('image/')) {
          const blob = await res.blob();
          const imgUrl = URL.createObjectURL(blob);
          UI.fileViewerContent.innerHTML = `<img src="${imgUrl}" style="max-width:100%; max-height:450px; object-fit:contain; border-radius:var(--radius-sm);" alt="preview">`;
        } else {
          const txt = await res.text();
          UI.fileViewerContent.textContent = txt;
        }
      } catch (err) {
        if (UI.fileViewerContent) UI.fileViewerContent.textContent = 'Error loading content: ' + err.message;
      }
    }

    if (UI.btnCloseFileViewer) {
      UI.btnCloseFileViewer.addEventListener('click', () => {
        if (UI.fileViewerDialog) UI.fileViewerDialog.classList.remove('show');
      });
    }
    if (UI.fileViewerDialog) {
      UI.fileViewerDialog.addEventListener('click', (e) => {
        if (e.target === UI.fileViewerDialog) UI.fileViewerDialog.classList.remove('show');
      });
    }

    // Sidebar Toggle
    function toggleSidebar(forceState) {
      if (!UI.sidebar) return;
      const isMobile = window.innerWidth <= 768;
      if (isMobile) {
        const isOpen = typeof forceState === 'boolean' ? forceState : !UI.sidebar.classList.contains('open');
        UI.sidebar.classList.toggle('open', isOpen);
        if (UI.sidebarBackdrop) UI.sidebarBackdrop.classList.toggle('show', isOpen);
      } else {
        // Desktop: cleanly toggle collapsed state, never show backdrop or blur screen
        if (UI.sidebarBackdrop) UI.sidebarBackdrop.classList.remove('show');
        const isCollapsed = typeof forceState === 'boolean' ? !forceState : (UI.sidebar.style.display !== 'none' && !UI.sidebar.classList.contains('collapsed'));
        if (isCollapsed) {
          UI.sidebar.classList.add('collapsed');
          UI.sidebar.style.display = 'none';
        } else {
          UI.sidebar.classList.remove('collapsed');
          UI.sidebar.style.display = 'flex';
        }
        localStorage.setItem('gufo_sidebar_collapsed', isCollapsed ? 'true' : 'false');
      }
    }

    if (window.innerWidth > 768 && localStorage.getItem('gufo_sidebar_collapsed') === 'true') {
      if (UI.sidebar) {
        UI.sidebar.classList.add('collapsed');
        UI.sidebar.style.display = 'none';
      }
    }

    if (UI.btnToggleSidebar) {
      UI.btnToggleSidebar.addEventListener('click', () => toggleSidebar());
    }
    if (UI.sidebarBackdrop) {
      UI.sidebarBackdrop.addEventListener('click', () => toggleSidebar(false));
    }
    if (UI.btnCloseSidebarMobile) {
      UI.btnCloseSidebarMobile.addEventListener('click', () => toggleSidebar(false));
    }

    // Model Live Telemetry & Monitoring
    async function refreshMonitorStats() {
      const headers = APP.token ? { 'Authorization': `Bearer ${APP.token}` } : {};
      try {
        const [monRes, slotsRes] = await Promise.all([
          fetch('/api/monitor', { headers }).catch(() => null),
          fetch('/slots', { headers }).catch(() => null)
        ]);

        if (monRes && monRes.ok) {
          const d = await monRes.json();
          if (UI.monValActiveMsg) UI.monValActiveMsg.textContent = d.messages_processing || 0;
          if (UI.monSubQueue) UI.monSubQueue.textContent = `Queue: ${d.messages_deferred || 0} deferred`;

          const genSpd = d.last_gen_speed > 0
            ? Number(d.last_gen_speed).toFixed(1)
            : (APP.lastSeenGenSpeed ? Number(APP.lastSeenGenSpeed).toFixed(1) : '0.0');
          if (UI.monValGenSpd) UI.monValGenSpd.textContent = genSpd;

          const promptSpd = d.last_prompt_speed > 0
            ? Number(d.last_prompt_speed).toFixed(1)
            : (APP.lastSeenPromptSpeed ? Number(APP.lastSeenPromptSpeed).toFixed(1) : '0.0');
          if (UI.monValPromptSpd) UI.monValPromptSpd.textContent = promptSpd;

          const genTok = d.last_gen_tokens || (APP.lastSeenGenTokens || 0);
          if (UI.monValGenTokens) UI.monValGenTokens.textContent = Number(genTok).toLocaleString();
          if (UI.monSubTotalGen) UI.monSubTotalGen.textContent = (d.total_gen_tokens || 0).toLocaleString();

          const promptTok = d.last_prompt_tokens || (APP.lastSeenPromptTokens || 0);
          if (UI.monValPromptTokens) UI.monValPromptTokens.textContent = Number(promptTok).toLocaleString();
          if (UI.monSubTotalPrompt) UI.monSubTotalPrompt.textContent = (d.total_prompt_tokens || 0).toLocaleString();
          if (UI.monSubCachedPrompt) UI.monSubCachedPrompt.textContent = (d.total_cached_prompt_tokens || 0).toLocaleString();

          if (d.total_draft_tokens > 0) {
            const acc = ((d.total_draft_accepted_tokens / d.total_draft_tokens) * 100).toFixed(1) + '%';
            if (UI.monValSpecAcc) UI.monValSpecAcc.textContent = acc;
            if (UI.monSubDrafts) UI.monSubDrafts.textContent = `Drafts: ${d.total_draft_accepted_tokens}/${d.total_draft_tokens} (${d.total_draft_rounds || 0} rds)`;
          } else {
            if (UI.monValSpecAcc) UI.monValSpecAcc.textContent = '100%';
            if (UI.monSubDrafts) UI.monSubDrafts.textContent = 'Standard Decode';
          }

          if (d.memory_status && UI.monSubMem) {
            UI.monSubMem.textContent = d.memory_status;
          }
        }

        if (slotsRes && slotsRes.ok) {
          const slots = await slotsRes.json();
          if (Array.isArray(slots)) {
            const busyCount = slots.filter(s => s.is_processing).length;
            if (UI.monSubSlots) UI.monSubSlots.textContent = `${slots.length - busyCount}/${slots.length} slots idle`;
            if (UI.monSlotsSummary) UI.monSlotsSummary.textContent = `${slots.length} total slot(s) &bull; ${busyCount} active`;

            if (UI.monSlotsList) {
              UI.monSlotsList.innerHTML = '';
              slots.forEach(slot => {
                const isBusy = slot.is_processing;
                const promptTokens = slot.n_prompt_tokens || 0;
                const cachedTokens = slot.n_prompt_tokens_cache || 0;
                const nextTok = (slot.next_token && slot.next_token[0]) || {};
                const decodedTokens = nextTok.n_decoded || 0;
                const maxCtx = slot.n_ctx || 4096;
                const usedCtx = promptTokens + decodedTokens;
                const ctxPct = Math.min(100, Math.round((usedCtx / maxCtx) * 100));

                const row = document.createElement('div');
                row.className = 'slot-row';
                row.innerHTML = `
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span class="slot-badge ${isBusy ? 'busy' : 'idle'}">
                      <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:${isBusy ? '#60a5fa' : '#71717a'};"></span>
                      Slot #${slot.id} ${isBusy ? 'Processing' : 'Idle'}
                    </span>
                    <span style="color:var(--text-dim); font-size:0.75rem;">${escapeHtml(slot.model || APP.modelId || 'gufo')}</span>
                  </div>
                  <div style="display:flex; align-items:center; gap:12px; font-family:var(--font-mono); font-size:0.75rem;">
                    <span>Prompt: <b style="color:var(--text);">${promptTokens}</b> <span style="color:var(--success); font-size:0.7rem;">(${cachedTokens} cached)</span></span>
                    <span>Decoded: <b style="color:var(--accent);">${decodedTokens}</b></span>
                    <span>Context: <b>${usedCtx}/${maxCtx}</b> (${ctxPct}%)</span>
                  </div>
                  <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${ctxPct}%;"></div>
                  </div>
                `;
                UI.monSlotsList.appendChild(row);
              });
            }
          }
        }
      } catch (e) {
        console.error('Failed to update monitor stats', e);
      }
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

    function toggleUserDropdown() {
      if (!UI.userPopupMenu) return;
      const isOpen = UI.userPopupMenu.style.display === 'flex';
      if (isOpen) {
        closeUserDropdown();
      } else {
        openUserDropdown();
      }
    }

    function openUserDropdown() {
      if (!UI.userPopupMenu) return;
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      if (UI.userPopupName) UI.userPopupName.textContent = username;
      if (UI.userPopupRoot) UI.userPopupRoot.textContent = `UserData/${username}`;
      if (UI.userMenuAuthTitle) {
        UI.userMenuAuthTitle.textContent = APP.user ? 'Sign Out' : 'Sign In / Register';
      }
      if (UI.userMenuAuthDesc) {
        UI.userMenuAuthDesc.textContent = APP.user ? `Logged in as ${APP.user.username}` : 'Multi-device sync & isolation';
      }
      if (UI.userMenuAuthIcon) {
        UI.userMenuAuthIcon.textContent = APP.user ? '🚪' : '🔑';
      }
      UI.userPopupMenu.style.display = 'flex';
    }

    function closeUserDropdown() {
      if (UI.userPopupMenu) UI.userPopupMenu.style.display = 'none';
    }

    if (UI.btnAuthUser) {
      UI.btnAuthUser.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleUserDropdown();
      });
    }

    if (UI.btnMenuFileBrowser) {
      UI.btnMenuFileBrowser.addEventListener('click', () => {
        closeUserDropdown();
        openUserFileBrowserModal();
      });
    }

    if (UI.btnMenuWorkdirSetup) {
      UI.btnMenuWorkdirSetup.addEventListener('click', () => {
        closeUserDropdown();
        openConvWorkdirModal();
      });
    }

    if (UI.btnMenuAuth) {
      UI.btnMenuAuth.addEventListener('click', () => {
        closeUserDropdown();
        if (APP.user) {
          if (confirm(`Logged in as ${APP.user.username}. Sign out?`)) {
            logout();
          }
        } else {
          openAuthModal('login');
        }
      });
    }

    if (UI.btnTopFileBrowser) {
      UI.btnTopFileBrowser.addEventListener('click', () => {
        openUserFileBrowserModal();
      });
    }

    document.addEventListener('click', (e) => {
      if (UI.userPopupMenu && UI.userPopupMenu.style.display === 'flex') {
        if (!e.target.closest('.user-menu-wrapper')) {
          closeUserDropdown();
        }
      }
    });
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

    // Settings Drawer (right-side dialog)
    function openDrawer(tab = 'inference') {
      switchDrawerTab(tab);
      if (tab === 'inference') {
        if (UI.setSystemPrompt) UI.setSystemPrompt.value = APP.settings.system_prompt;
        if (UI.rangeTemp) UI.rangeTemp.value = APP.settings.temperature;
        if (UI.valTemp) UI.valTemp.textContent = Number(APP.settings.temperature).toFixed(2);
        if (UI.rangeTopP) UI.rangeTopP.value = APP.settings.top_p;
        if (UI.valTopP) UI.valTopP.textContent = Number(APP.settings.top_p).toFixed(2);
        if (UI.inpMaxTokens) UI.inpMaxTokens.value = APP.settings.max_tokens;
        renderSettingsPromptList();
      } else {
        loadTools();
      }
      if (UI.drawerBackdrop) UI.drawerBackdrop.classList.add('show');
      if (UI.drawerPanel) UI.drawerPanel.classList.add('show');
    }

    function closeDrawer() {
      if (UI.drawerBackdrop) UI.drawerBackdrop.classList.remove('show');
      if (UI.drawerPanel) UI.drawerPanel.classList.remove('show');
    }

    function switchDrawerTab(tab) {
      if (UI.drawerTabInference) UI.drawerTabInference.classList.toggle('active', tab === 'inference');
      if (UI.drawerTabTools) UI.drawerTabTools.classList.toggle('active', tab === 'tools');
      if (UI.drawerPaneInference) UI.drawerPaneInference.style.display = (tab === 'inference' ? 'flex' : 'none');
      if (UI.drawerPaneTools) UI.drawerPaneTools.style.display = (tab === 'tools' ? 'flex' : 'none');
      if (tab === 'inference') renderSettingsPromptList();
    }

    if (UI.drawerTabInference) UI.drawerTabInference.addEventListener('click', () => switchDrawerTab('inference'));
    if (UI.drawerTabTools) UI.drawerTabTools.addEventListener('click', () => {
      switchDrawerTab('tools');
      loadTools();
    });

    if (UI.btnOpenDrawer) UI.btnOpenDrawer.addEventListener('click', () => openDrawer('inference'));
    if (UI.btnOpenTools) UI.btnOpenTools.addEventListener('click', () => openDrawer('tools'));
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

    // Attachment Popover Menu Logic
    function toggleAttachPopup() {
      if (!UI.attachPopupMenu) return;
      const isOpen = UI.attachPopupMenu.style.display === 'flex';
      if (isOpen) {
        closeAttachPopup();
      } else {
        openAttachPopup();
      }
    }

    // 7 Attachment & Context Menu Items Implementation
    const TOOL_GROUPS = [
      { id: 'fs', name: 'Filesystem Tools', icon: '📁', tools: ['read_file', 'write_file', 'edit_file', 'apply_diff'] },
      { id: 'search', name: 'Code & Inspection', icon: '🔍', tools: ['file_glob_search', 'grep_search', 'get_datetime'] },
      { id: 'shell', name: 'Shell Execution', icon: '💻', tools: ['exec_shell_command'] },
      { id: 'web', name: 'Web Access', icon: '🌐', tools: ['fetch_web_page'] },
      { id: 'mcp_custom', name: 'MCP & Custom Tools', icon: '🔌', isCustom: true }
    ];

    const DEFAULT_SYSTEM_PROMPTS = {
      default: 'You are a helpful, respectful and honest assistant.',
      coder: 'You are an expert software engineer and systems architect. Write clean, idiomatic, high-performance code with minimal fluff and direct explanations.',
      concise: 'You are a concise, direct AI assistant. Keep responses brief, clear, and straight to the point without filler or preamble.',
      reasoner: 'You are a meticulous reasoning assistant. Think through problems step by step, exploring assumptions, edge cases, and verifying conclusions.',
      creative: 'You are an imaginative creative assistant and brainstormer, providing novel ideas, rich analogies, and diverse perspectives.'
    };

    const PROMPT_PRESETS = [
      { key: 'default', name: 'Gufo Assistant', tag: 'Balanced helper', icon: '🦉', desc: 'Helpful, respectful, and honest everyday AI assistant' },
      { key: 'coder', name: 'Code Expert', tag: 'Architecture & code', icon: '💻', desc: 'Expert software engineer and systems architect, clean high-performance code' },
      { key: 'concise', name: 'Concise & Direct', tag: 'No fluff', icon: '⚡', desc: 'Brief, clear, and straight to the point without filler or preamble' },
      { key: 'reasoner', name: 'Deep Reasoner', tag: 'Logic & analysis', icon: '🧠', desc: 'Meticulous reasoning assistant, step-by-step logic and verification' },
      { key: 'creative', name: 'Creative Writer', tag: 'Brainstorm & ideas', icon: '🎨', desc: 'Imaginative creative assistant, brainstormer, rich analogies' }
    ];

    function getActivePromptKey() {
      const current = (APP.settings && APP.settings.system_prompt) ? APP.settings.system_prompt : '';
      for (const p of PROMPT_PRESETS) {
        if (DEFAULT_SYSTEM_PROMPTS[p.key] === current) return p.key;
      }
      if (APP.customPrompts) {
        for (let i = 0; i < APP.customPrompts.length; i++) {
          if (APP.customPrompts[i].prompt === current) return `custom_${i}`;
        }
      }
      return current ? 'active_custom' : 'default';
    }

    function getActivePromptMeta() {
      const key = getActivePromptKey();
      const preset = PROMPT_PRESETS.find(p => p.key === key);
      if (preset) return { key: preset.key, name: preset.name, icon: preset.icon };
      if (key.startsWith('custom_')) {
        const idx = parseInt(key.split('_')[1], 10);
        if (APP.customPrompts && APP.customPrompts[idx]) {
          return { key, name: APP.customPrompts[idx].name, icon: '⭐' };
        }
      }
      return { key: 'custom', name: 'Custom', icon: '🎭' };
    }

    function openNewPromptModal() {
      closeAttachPopup();
      if (UI.newPromptNameInp) UI.newPromptNameInp.value = '';
      if (UI.newPromptTextInp) UI.newPromptTextInp.value = '';
      if (UI.editPromptIdx) UI.editPromptIdx.value = '';
      if (UI.newPromptDialogTitle) UI.newPromptDialogTitle.textContent = 'Add Custom System Prompt';
      if (UI.newPromptDialog) UI.newPromptDialog.classList.add('show');
    }

    function openEditPromptModal(promptObj) {
      closeAttachPopup();
      if (UI.newPromptNameInp) UI.newPromptNameInp.value = promptObj.name || '';
      if (UI.newPromptTextInp) UI.newPromptTextInp.value = promptObj.prompt || '';
      if (UI.editPromptIdx) UI.editPromptIdx.value = promptObj.isDefault ? `preset_${promptObj.key}` : String(promptObj.customIdx);
      if (UI.newPromptDialogTitle) {
        UI.newPromptDialogTitle.textContent = promptObj.isDefault ? `Duplicate & Edit "${promptObj.name}"` : `Edit "${promptObj.name}"`;
      }
      if (UI.newPromptDialog) UI.newPromptDialog.classList.add('show');
    }

    async function deleteCustomPrompt(idx, name) {
      if (!confirm(`Delete custom prompt "${name}"? This cannot be undone.`)) return;
      if (idx >= 0 && idx < APP.customPrompts.length) {
        const deleted = APP.customPrompts.splice(idx, 1)[0];
        localStorage.setItem('gufo_custom_prompts', JSON.stringify(APP.customPrompts));
        if (APP.settings.system_prompt === deleted.prompt) {
          await selectSystemPrompt('default');
        } else {
          renderSystemPromptDropdown();
          calculateContextSize();
          updateInputMetaBar();
          if (!APP.activeConv || !APP.activeConv.messages || APP.activeConv.messages.length === 0) {
            renderHeroDashboard();
          }
        }
        renderSettingsPromptList();
      }
    }

    function renderSettingsPromptList() {
      if (!UI.settingsPromptList) return;
      UI.settingsPromptList.innerHTML = '';

      const activeKey = getActivePromptKey();
      const currentPromptText = APP.settings.system_prompt || '';

      if (UI.settingsActivePromptTag) {
        const meta = getActivePromptMeta();
        UI.settingsActivePromptTag.textContent = `${meta.icon} ${meta.name}`;
      }
      if (UI.setSystemPrompt && document.activeElement !== UI.setSystemPrompt) {
        UI.setSystemPrompt.value = currentPromptText;
      }

      const allPrompts = [];

      // 1. Standard Presets
      PROMPT_PRESETS.forEach(p => {
        allPrompts.push({
          key: p.key,
          name: p.name,
          icon: p.icon,
          tag: p.tag,
          desc: p.desc,
          prompt: DEFAULT_SYSTEM_PROMPTS[p.key] || '',
          isDefault: true
        });
      });

      // 2. Custom Presets
      if (APP.customPrompts && APP.customPrompts.length > 0) {
        APP.customPrompts.forEach((cp, idx) => {
          allPrompts.push({
            key: `custom_${idx}`,
            name: cp.name,
            icon: '⭐',
            tag: 'Custom Persona',
            desc: cp.prompt.slice(0, 100),
            prompt: cp.prompt,
            isDefault: false,
            customIdx: idx
          });
        });
      }

      if (UI.settingsPromptsCount) {
        UI.settingsPromptsCount.textContent = `${allPrompts.length} prompts`;
      }

      allPrompts.forEach(p => {
        const isActive = (p.key === activeKey) || (p.prompt === currentPromptText);
        const card = document.createElement('div');
        card.className = `settings-prompt-item ${isActive ? 'active' : ''}`;

        card.innerHTML = `
          <div class="settings-prompt-item-header">
            <div style="display:flex; align-items:center; gap:8px; min-width:0;">
              <span class="settings-prompt-icon">${p.icon}</span>
              <div style="min-width:0;">
                <div style="display:flex; align-items:center; gap:6px;">
                  <span class="settings-prompt-title">${escapeHtml(p.name)}</span>
                  ${p.isDefault ? '<span class="tag-pill" style="font-size:0.6rem;">Preset</span>' : '<span class="tag-pill" style="font-size:0.6rem; color:var(--accent); border-color:rgba(124,58,237,0.3);">Custom</span>'}
                  ${isActive ? '<span class="hero-card-badge" style="background:rgba(34,197,94,0.15); color:#86efac; border-color:rgba(34,197,94,0.3); font-size:0.6rem;">✓ Active</span>' : ''}
                </div>
                <div class="settings-prompt-desc">${escapeHtml(p.tag || '')}</div>
              </div>
            </div>
            <div class="settings-prompt-actions">
              ${!isActive ? `<button class="btn-top btn-prompt-use" type="button" title="Set as active system prompt" style="padding:2px 8px; font-size:0.7rem; color:var(--text);">Use</button>` : ''}
              <button class="btn-top btn-prompt-edit" type="button" title="Edit prompt" style="padding:2px 6px; font-size:0.75rem;">✏️</button>
              ${!p.isDefault ? `<button class="btn-top btn-prompt-del" type="button" title="Delete custom prompt" style="padding:2px 6px; font-size:0.85rem; line-height:1; color:var(--danger); border-color:rgba(239,68,68,0.25);">&times;</button>` : ''}
            </div>
          </div>
          <div class="settings-prompt-preview">${escapeHtml(p.prompt)}</div>
        `;

        const useBtn = card.querySelector('.btn-prompt-use');
        if (useBtn) {
          useBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            selectSystemPrompt(p.key);
            renderSettingsPromptList();
          });
        }

        const editBtn = card.querySelector('.btn-prompt-edit');
        if (editBtn) {
          editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openEditPromptModal(p);
          });
        }

        const delBtn = card.querySelector('.btn-prompt-del');
        if (delBtn) {
          delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteCustomPrompt(p.customIdx, p.name);
          });
        }

        UI.settingsPromptList.appendChild(card);
      });
    }

    function updateHeroWorkdirDisplay() {
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      const userPrefix = `UserData/${username}/`;
      let wdName = `UserData/${username}/...`;
      if (APP.activeConv && APP.activeConv.working_directory) {
        let wd = APP.activeConv.working_directory;
        if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);
        wdName = `UserData/${username}/${wd}`;
      } else if (APP.newChatWorkdir) {
        let wd = APP.newChatWorkdir;
        if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);
        wdName = `UserData/${username}/${wd}`;
      }
      const el = document.getElementById('hero-workdir-sub');
      if (el) el.textContent = wdName;
      updateAttachPopupUI();
    }

    function updateInputMetaBar() {
      const pMeta = getActivePromptMeta();
      const elPrompt = document.getElementById('input-meta-prompt');
      if (elPrompt) {
        elPrompt.textContent = `${pMeta.icon} ${pMeta.name}`;
        elPrompt.title = `System prompt: ${pMeta.name} (click to cycle)`;
      }

      const elThinking = document.getElementById('input-meta-thinking');
      if (elThinking) {
        const lvl = APP.thinkingLevel || 'auto';
        const displayLvl = lvl.charAt(0).toUpperCase() + lvl.slice(1);
        elThinking.textContent = `🧠Think: ${displayLvl}`;
        elThinking.title = `Thinking level: ${lvl} (click to cycle)`;
      }

      const elPerm = document.getElementById('input-meta-perm');
      if (elPerm) {
        const isAllow = APP.toolPermission === 'allow';
        elPerm.textContent = isAllow ? '⚡ Allow all' : '🛡️ Ask';
        elPerm.title = isAllow ? 'Tools execute autonomously without asking (click to toggle)' : 'Ask approval before executing tools (click to toggle)';
      }

      calculateContextSize();
    }

    async function selectSystemPrompt(key) {
      let promptText = '';
      if (DEFAULT_SYSTEM_PROMPTS[key]) {
        promptText = DEFAULT_SYSTEM_PROMPTS[key];
      } else if (key.startsWith('custom_')) {
        const idx = parseInt(key.split('_')[1], 10);
        if (APP.customPrompts && APP.customPrompts[idx]) {
          promptText = APP.customPrompts[idx].prompt;
        }
      }
      if (!promptText) return;

      APP.settings.system_prompt = promptText;
      if (UI.setSystemPrompt) UI.setSystemPrompt.value = promptText;
      if (UI.popupSystemPromptSelect) {
        UI.popupSystemPromptSelect.value = key;
      }

      if (APP.activeId) {
        try {
          await apiFetch(`/api/conversations/${APP.activeId}`, {
            method: 'PATCH',
            body: { system_prompt: promptText }
          });
          if (APP.activeConv) APP.activeConv.system_prompt = promptText;
        } catch (_) {}
      }

      try {
        await apiFetch('/api/settings', { method: 'POST', body: APP.settings });
      } catch (_) {}

      const activeMeta = getActivePromptMeta();
      const heroPromptBadge = document.getElementById('hero-prompt-badge');
      if (heroPromptBadge) {
        heroPromptBadge.textContent = `${activeMeta.icon} ${activeMeta.name}`;
      }
      const promptThumbs = document.querySelectorAll('.hero-prompt-thumb');
      promptThumbs.forEach(t => {
        t.classList.toggle('active', t.dataset.key === key);
      });

      calculateContextSize();
      updateInputMetaBar();
      renderSettingsPromptList();
    }

    function cycleSystemPrompt() {
      const curKey = getActivePromptKey();
      const keys = PROMPT_PRESETS.map(p => p.key);
      if (APP.customPrompts) {
        for (let i = 0; i < APP.customPrompts.length; i++) keys.push(`custom_${i}`);
      }
      const curIdx = keys.indexOf(curKey);
      const nextIdx = (curIdx + 1) % keys.length;
      selectSystemPrompt(keys[nextIdx]);
    }

    function openAttachPopup() {
      if (!UI.attachPopupMenu) return;
      updateAttachPopupUI();
      UI.attachPopupMenu.style.display = 'flex';
    }

    function closeAttachPopup() {
      if (UI.attachPopupMenu) UI.attachPopupMenu.style.display = 'none';
      if (UI.attachToolsGroupsSub) UI.attachToolsGroupsSub.style.display = 'none';
      if (UI.btnToggleToolsSubmenu) UI.btnToggleToolsSubmenu.textContent = 'Groups ▼';
    }

    function updateAttachPopupUI() {
      // 3. Working directory display
      if (UI.attachPopupWorkdirSub) {
        const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
        let wd = (APP.activeConv && APP.activeConv.working_directory) ? APP.activeConv.working_directory : '';
        const userPrefix = `UserData/${username}/`;
        if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);
        if (wd) {
          UI.attachPopupWorkdirSub.textContent = `UserData/${username}/${wd}`;
          UI.attachPopupWorkdirSub.title = `UserData/${username}/${wd}`;
        } else {
          UI.attachPopupWorkdirSub.textContent = `UserData/${username}/...`;
          UI.attachPopupWorkdirSub.title = 'Default chat folder';
        }
      }

      // 4. Master tools switch & status
      if (UI.chkMasterTools) {
        UI.chkMasterTools.checked = APP.toolsEnabled;
      }
      const activeInbuilt = (APP.tools.inbuilt || []).filter(t => t.name !== 'ask_user' && t.enabled).length;
      const activeCustom = (APP.tools.customTools || []).filter(t => t.enabled).length;
      const total = activeInbuilt + activeCustom;
      if (UI.attachToolsStatusText) {
        UI.attachToolsStatusText.textContent = APP.toolsEnabled ? `${total} tools active` : 'Tools disabled';
      }

      // 4.a Render tool groups
      renderToolGroupsSubmenu();

      // 5. System prompt dropdown
      renderSystemPromptDropdown();

      // 6. Thinking level buttons
      renderThinkingLevelButtons();

      // Tool permission buttons
      renderToolPermissionButtons();

      // 7. Context size calculation
      calculateContextSize();
    }

    function updateWorkingDirUI() {
      updateAttachPopupUI();
    }

    function renderToolGroupsSubmenu() {
      if (!UI.attachToolGroupsList) return;
      UI.attachToolGroupsList.innerHTML = '';

      TOOL_GROUPS.forEach(group => {
        let totalInGroup = 0;
        let activeInGroup = 0;

        if (group.isCustom) {
          totalInGroup = (APP.tools.customTools || []).length + (APP.tools.mcpServers || []).length;
          activeInGroup = (APP.tools.customTools || []).filter(c => c.enabled).length + (APP.tools.mcpServers || []).filter(s => s.enabled).length;
        } else {
          const groupTools = (APP.tools.inbuilt || []).filter(t => group.tools.includes(t.name));
          totalInGroup = groupTools.length;
          activeInGroup = groupTools.filter(t => t.enabled).length;
        }

        const isGroupEnabled = totalInGroup > 0 ? activeInGroup > 0 : true;

        const row = document.createElement('div');
        row.className = 'attach-group-row';
        row.innerHTML = `
          <div class="attach-group-info">
            <span>${group.icon}</span>
            <div>
              <div class="attach-group-title">${escapeHtml(group.name)}</div>
              <div class="attach-group-count">${activeInGroup}/${totalInGroup} active</div>
            </div>
          </div>
          <label class="switch-toggle-mini" title="Toggle ${escapeHtml(group.name)}" onclick="event.stopPropagation();">
            <input type="checkbox" ${isGroupEnabled ? 'checked' : ''} ${!APP.toolsEnabled ? 'disabled' : ''}>
            <span class="slider-mini"></span>
          </label>
        `;

        const chk = row.querySelector('input[type="checkbox"]');
        chk.addEventListener('change', (e) => {
          const enable = e.target.checked;
          if (group.isCustom) {
            (APP.tools.customTools || []).forEach(c => c.enabled = enable);
            (APP.tools.mcpServers || []).forEach(s => s.enabled = enable);
          } else {
            (APP.tools.inbuilt || []).forEach(t => {
              if (group.tools.includes(t.name)) {
                t.enabled = enable;
              }
            });
            saveInbuiltTools();
          }
          updateToolsPill();
          renderToolGroupsSubmenu();
        });

        UI.attachToolGroupsList.appendChild(row);
      });
    }

    function renderSystemPromptDropdown() {
      if (!UI.popupSystemPromptSelect) return;
      const currentPrompt = APP.settings.system_prompt || '';
      UI.popupSystemPromptSelect.innerHTML = '';

      let matched = false;
      PROMPT_PRESETS.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.key;
        opt.textContent = `${p.icon} ${p.name}`;
        if (DEFAULT_SYSTEM_PROMPTS[p.key] === currentPrompt) {
          opt.selected = true;
          matched = true;
        }
        UI.popupSystemPromptSelect.appendChild(opt);
      });

      // User custom prompts
      if (APP.customPrompts && APP.customPrompts.length > 0) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = 'Custom Presets';
        APP.customPrompts.forEach((cp, idx) => {
          const opt = document.createElement('option');
          opt.value = `custom_${idx}`;
          opt.textContent = `⭐ ${cp.name}`;
          if (cp.prompt === currentPrompt) {
            opt.selected = true;
            matched = true;
          }
          optgroup.appendChild(opt);
        });
        UI.popupSystemPromptSelect.appendChild(optgroup);
      }

      // If current prompt doesn't match presets, add custom option
      if (!matched && currentPrompt) {
        const customOpt = document.createElement('option');
        customOpt.value = 'active_custom';
        customOpt.textContent = 'Active Custom Prompt';
        customOpt.selected = true;
        UI.popupSystemPromptSelect.appendChild(customOpt);
      }
    }

    if (UI.popupSystemPromptSelect) {
      UI.popupSystemPromptSelect.addEventListener('change', (e) => {
        selectSystemPrompt(e.target.value);
      });
    }

    // New Custom System Prompt Dialog handlers
    if (UI.btnPopupNewPrompt) {
      UI.btnPopupNewPrompt.addEventListener('click', openNewPromptModal);
    }

    function closeNewPromptModal() {
      if (UI.newPromptDialog) UI.newPromptDialog.classList.remove('show');
    }

    if (UI.btnCloseNewPrompt) UI.btnCloseNewPrompt.addEventListener('click', closeNewPromptModal);
    if (UI.btnCancelNewPrompt) UI.btnCancelNewPrompt.addEventListener('click', closeNewPromptModal);
    if (UI.newPromptDialog) {
      UI.newPromptDialog.addEventListener('click', e => {
        if (e.target === UI.newPromptDialog) closeNewPromptModal();
      });
    }

    if (UI.btnSaveNewPrompt) {
      UI.btnSaveNewPrompt.addEventListener('click', async () => {
        const name = (UI.newPromptNameInp ? UI.newPromptNameInp.value : '').trim();
        const text = (UI.newPromptTextInp ? UI.newPromptTextInp.value : '').trim();
        if (!name || !text) {
          alert('Please provide both a prompt name and prompt instructions.');
          return;
        }

        const editVal = UI.editPromptIdx ? UI.editPromptIdx.value : '';
        let targetKey = '';

        if (editVal !== '' && !editVal.startsWith('preset_')) {
          const idx = parseInt(editVal, 10);
          if (APP.customPrompts[idx]) {
            APP.customPrompts[idx] = { name, prompt: text };
            targetKey = `custom_${idx}`;
          }
        } else {
          APP.customPrompts.push({ name, prompt: text });
          targetKey = `custom_${APP.customPrompts.length - 1}`;
        }

        localStorage.setItem('gufo_custom_prompts', JSON.stringify(APP.customPrompts));
        closeNewPromptModal();
        renderSystemPromptDropdown();
        await selectSystemPrompt(targetKey);
        renderSettingsPromptList();
        if (!APP.activeConv || !APP.activeConv.messages || APP.activeConv.messages.length === 0) {
          renderHeroDashboard();
        }
      });
    }

    if (UI.btnSettingsAddPrompt) {
      UI.btnSettingsAddPrompt.addEventListener('click', openNewPromptModal);
    }

    if (UI.btnSaveActivePromptText) {
      UI.btnSaveActivePromptText.addEventListener('click', async () => {
        const text = (UI.setSystemPrompt ? UI.setSystemPrompt.value : '').trim();
        if (!text) {
          alert('System prompt cannot be empty.');
          return;
        }
        APP.settings.system_prompt = text;
        if (APP.activeId) {
          try {
            await apiFetch(`/api/conversations/${APP.activeId}`, {
              method: 'PATCH',
              body: { system_prompt: text }
            });
            if (APP.activeConv) APP.activeConv.system_prompt = text;
          } catch (_) {}
        }
        try {
          await apiFetch('/api/settings', { method: 'POST', body: APP.settings });
        } catch (_) {}

        calculateContextSize();
        updateInputMetaBar();
        renderSettingsPromptList();
        renderSystemPromptDropdown();
        if (!APP.activeConv || !APP.activeConv.messages || APP.activeConv.messages.length === 0) {
          renderHeroDashboard();
        }
        alert('Active system prompt updated!');
      });
    }

    // 6. Thinking Level Control (auto, minimum, low, medium, high, xhigh)
    function setThinkingLevel(level) {
      if (!['auto', 'minimum', 'low', 'medium', 'high', 'xhigh'].includes(level)) level = 'auto';
      APP.thinkingLevel = level;
      localStorage.setItem('gufo_thinking_level', APP.thinkingLevel);
      renderThinkingLevelButtons();
      const heroThinkBadge = document.getElementById('hero-think-badge');
      if (heroThinkBadge) {
        heroThinkBadge.textContent = `${level} reasoning`;
      }
      updateInputMetaBar();
    }

    function cycleThinkingLevel() {
      const levels = ['auto', 'minimum', 'low', 'medium', 'high', 'xhigh'];
      const curIdx = levels.indexOf(APP.thinkingLevel);
      const nextIdx = (curIdx + 1) % levels.length;
      setThinkingLevel(levels[nextIdx]);
    }

    function renderThinkingLevelButtons() {
      const btns = document.querySelectorAll('#thinking-level-control .seg-btn, .hero-think-btn');
      btns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.level === APP.thinkingLevel);
      });
    }

    if (UI.thinkingLevelControl) {
      UI.thinkingLevelControl.addEventListener('click', (e) => {
        const btn = e.target.closest('.seg-btn');
        if (!btn) return;
        setThinkingLevel(btn.dataset.level || 'auto');
      });
    }

    // Tool Permission Control (Ask before running / Allow all)
    function setToolPermission(perm) {
      if (!['ask', 'allow'].includes(perm)) perm = 'ask';
      APP.toolPermission = perm;
      localStorage.setItem('gufo_tool_permission', APP.toolPermission);
      renderToolPermissionButtons();
      const heroPermBadge = document.getElementById('hero-perm-badge');
      if (heroPermBadge) {
        heroPermBadge.textContent = perm === 'allow' ? 'Allow all (autonomous)' : 'Ask before running';
      }
      updateInputMetaBar();
    }

    function cycleToolPermission() {
      setToolPermission(APP.toolPermission === 'allow' ? 'ask' : 'allow');
    }

    function renderToolPermissionButtons() {
      const btns = document.querySelectorAll('#tool-perm-control .seg-btn, .hero-perm-btn');
      btns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.perm === APP.toolPermission);
      });
      const popupPermLabel = document.getElementById('popup-perm-label');
      if (popupPermLabel) {
        popupPermLabel.textContent = APP.toolPermission === 'allow' ? 'Allow all' : 'Ask';
      }
    }

    if (UI.toolPermControl) {
      UI.toolPermControl.addEventListener('click', (e) => {
        const btn = e.target.closest('.seg-btn');
        if (!btn) return;
        setToolPermission(btn.dataset.perm || 'ask');
      });
    }

    // 7. Context Size Calculation & Compaction
    function calculateContextSize() {
      let chars = (APP.settings.system_prompt || '').length;
      if (APP.activeConv && APP.activeConv.messages) {
        for (const m of APP.activeConv.messages) {
          chars += (m.content || '').length + (m.reasoning_content || '').length + 24;
        }
      }
      if (APP.attachedFiles) {
        for (const f of APP.attachedFiles) {
          chars += (f.content || '').length + 60;
        }
      }
      const estimatedTokens = Math.max(0, Math.round(chars / 3.8));
      const maxContext = 262144; // 262k
      const pct = Math.min(100, (estimatedTokens / maxContext) * 100);

      if (UI.popupContextTokens) {
        UI.popupContextTokens.textContent = `${estimatedTokens.toLocaleString()} / 262k`;
      }
      if (UI.popupContextBar) {
        UI.popupContextBar.style.width = `${Math.max(1, pct)}%`;
      }
      if (UI.popupContextPct) {
        UI.popupContextPct.textContent = `${pct.toFixed(1)}% used`;
      }
      const elCtx = document.getElementById('input-meta-ctx');
      if (elCtx) {
        elCtx.textContent = `📊 ${estimatedTokens.toLocaleString()} ctx (${pct.toFixed(1)}%)`;
        elCtx.title = `Estimated context: ${estimatedTokens.toLocaleString()} / 262k tokens (${pct.toFixed(1)}%)`;
      }
      return { estimatedTokens, maxContext, pct };
    }

    async function compactActiveConversation() {
      if (!APP.activeId || !APP.activeConv || !APP.activeConv.messages || APP.activeConv.messages.length <= 3) {
        alert('Conversation context is already minimal; compaction not required.');
        return;
      }
      const msgs = APP.activeConv.messages;
      if (!confirm(`Compact this conversation? Older turns will be condensed into a concise summary to free up model context.`)) {
        return;
      }

      if (UI.btnCompactContext) {
        UI.btnCompactContext.disabled = true;
        UI.btnCompactContext.textContent = 'Compacting...';
      }

      try {
        const keepLast = 4;
        const olderMsgs = msgs.slice(0, msgs.length - keepLast);
        const recentMsgs = msgs.slice(msgs.length - keepLast);

        let summaryText = 'Summary of earlier conversation:\n';
        for (const m of olderMsgs) {
          const snippet = (m.content || '').replace(/\s+/g, ' ').slice(0, 140);
          summaryText += `- ${m.role}: ${snippet}${m.content.length > 140 ? '...' : ''}\n`;
        }

        const compactedMessages = [
          {
            role: 'system',
            content: `[Previous Context Summary]\n${summaryText}`
          },
          ...recentMsgs
        ];

        const res = await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
          method: 'PUT',
          body: { messages: compactedMessages }
        });

        if (res.ok) {
          const refreshRes = await apiFetch(`/api/conversations/${APP.activeId}`);
          if (refreshRes.ok) {
            APP.activeConv = await refreshRes.json();
            renderMessages(APP.activeConv.messages);
          }
          calculateContextSize();
          closeAttachPopup();
          alert('Context compacted successfully!');
        } else {
          throw new Error('Failed to save compacted messages');
        }
      } catch (err) {
        alert('Compaction failed: ' + err.message);
      } finally {
        if (UI.btnCompactContext) {
          UI.btnCompactContext.disabled = false;
          UI.btnCompactContext.textContent = '🧹 Compact';
        }
      }
    }

    if (UI.btnCompactContext) {
      UI.btnCompactContext.addEventListener('click', compactActiveConversation);
    }

    // 1. Attach Images Item
    if (UI.attachItemImage) {
      UI.attachItemImage.addEventListener('click', () => {
        closeAttachPopup();
        if (UI.fileUploaderImage) UI.fileUploaderImage.click();
      });
    }

    // 2. Attach Files Item
    if (UI.attachItemFiles) {
      UI.attachItemFiles.addEventListener('click', () => {
        closeAttachPopup();
        if (UI.fileUploaderDoc) UI.fileUploaderDoc.click();
      });
    }

    // 3. Working Directory Item
    if (UI.attachItemWorkdir) {
      UI.attachItemWorkdir.addEventListener('click', () => {
        closeAttachPopup();
        openConvWorkdirModal();
      });
    }

    // 4. Toggle Tool Groups Submenu Button
    if (UI.btnToggleToolsSubmenu) {
      UI.btnToggleToolsSubmenu.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!UI.attachToolsGroupsSub) return;
        const isHidden = UI.attachToolsGroupsSub.style.display === 'none';
        UI.attachToolsGroupsSub.style.display = isHidden ? 'flex' : 'none';
        UI.btnToggleToolsSubmenu.textContent = isHidden ? 'Groups ▲' : 'Groups ▼';
      });
    }

    if (UI.attachBtnManageTools) {
      UI.attachBtnManageTools.addEventListener('click', () => {
        closeAttachPopup();
        openDrawer('tools');
      });
    }

    // Attachment Button Toggle
    if (UI.btnUploadFile) {
      UI.btnUploadFile.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleAttachPopup();
      });
    }

    document.addEventListener('click', (e) => {
      if (UI.attachPopupMenu && UI.attachPopupMenu.style.display === 'flex') {
        if (!e.target.closest('.attach-menu-container') && !e.target.closest('#new-prompt-dialog') && !e.target.closest('.hero-attach-card')) {
          closeAttachPopup();
        }
      }
    });

    // Conversation-Based Working Directory Modal
    async function openConvWorkdirModal() {
      if (!UI.convWorkdirDialog) return;
      const conv = APP.activeConv || (APP.conversations.length > 0 ? APP.conversations[0] : null);
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      const userPrefix = `UserData/${username}/`;

      if (UI.convWorkdirActiveTitle) {
        UI.convWorkdirActiveTitle.textContent = (conv && conv.title) ? conv.title : 'New Chat';
      }
      if (UI.convWorkdirActiveId) {
        UI.convWorkdirActiveId.textContent = conv ? `ID: ${conv.id}` : '';
      }
      if (UI.convWorkdirUserPrefix) UI.convWorkdirUserPrefix.textContent = userPrefix;
      if (UI.convWorkdirInpPrefix) UI.convWorkdirInpPrefix.textContent = userPrefix;

      let currentRelFolder = '';
      if (conv && conv.working_directory) {
        let wd = conv.working_directory;
        if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);
        currentRelFolder = wd;
      } else if (conv) {
        currentRelFolder = `${conv.id} - ${conv.title || 'Chat'}`;
      }

      if (UI.convWorkdirInp) {
        UI.convWorkdirInp.value = currentRelFolder;
      }
      if (UI.convWorkdirDisplayBadge) {
        UI.convWorkdirDisplayBadge.textContent = userPrefix + (currentRelFolder || '');
      }
      if (UI.convWorkdirDefaultHint && conv) {
        UI.convWorkdirDefaultHint.textContent = `${conv.id} - ${conv.title || 'Chat'}`;
      }

      // Populate dropdown with all folders under username
      if (UI.convWorkdirSelect) {
        UI.convWorkdirSelect.innerHTML = '<option value="">-- Choose existing folder under username --</option>';
        try {
          const res = await apiFetch('/api/user-files');
          if (res.ok) {
            const data = await res.json();
            const folders = data.folders_list || [];
            folders.sort();
            folders.forEach(f => {
              const opt = document.createElement('option');
              opt.value = f;
              opt.textContent = '📁 ' + f;
              if (f === currentRelFolder) opt.selected = true;
              UI.convWorkdirSelect.appendChild(opt);
            });
          }
        } catch (_) {}
      }

      UI.convWorkdirDialog.classList.add('show');
    }

    function closeConvWorkdirModal() {
      if (UI.convWorkdirDialog) UI.convWorkdirDialog.classList.remove('show');
    }

    if (UI.btnCloseConvWorkdir) UI.btnCloseConvWorkdir.addEventListener('click', closeConvWorkdirModal);
    if (UI.btnConvWorkdirCancel) UI.btnConvWorkdirCancel.addEventListener('click', closeConvWorkdirModal);
    if (UI.convWorkdirDialog) {
      UI.convWorkdirDialog.addEventListener('click', (e) => {
        if (e.target === UI.convWorkdirDialog) closeConvWorkdirModal();
      });
    }

    if (UI.convWorkdirSelect) {
      UI.convWorkdirSelect.addEventListener('change', (e) => {
        if (e.target.value) {
          if (UI.convWorkdirInp) UI.convWorkdirInp.value = e.target.value;
          const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
          if (UI.convWorkdirDisplayBadge) {
            UI.convWorkdirDisplayBadge.textContent = `UserData/${username}/${e.target.value}`;
          }
        }
      });
    }

    if (UI.convWorkdirInp) {
      UI.convWorkdirInp.addEventListener('input', (e) => {
        const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
        if (UI.convWorkdirDisplayBadge) {
          UI.convWorkdirDisplayBadge.textContent = `UserData/${username}/${e.target.value.trim()}`;
        }
      });
    }

    if (UI.btnConvWorkdirOpenBrowser) {
      UI.btnConvWorkdirOpenBrowser.addEventListener('click', () => {
        const targetPath = UI.convWorkdirInp ? UI.convWorkdirInp.value.trim() : '';
        closeConvWorkdirModal();
        openUserFileBrowserModal(targetPath);
      });
    }

    if (UI.btnConvWorkdirReset) {
      UI.btnConvWorkdirReset.addEventListener('click', async () => {
        if (!APP.activeId) {
          const conv = APP.activeConv || (APP.conversations.length > 0 ? APP.conversations[0] : null);
          if (conv) {
            APP.activeId = conv.id;
            APP.activeConv = conv;
          }
        }
        if (APP.activeId) {
          try {
            const res = await apiFetch(`/api/conversations/${APP.activeId}`, {
              method: 'PATCH',
              body: { working_directory: 'default' }
            });
            if (res.ok) {
              const data = await res.json();
              const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
              const userPrefix = `UserData/${username}/`;
              let wd = data.working_directory || '';
              if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);

              if (APP.activeConv) APP.activeConv.working_directory = wd;
              const targetConv = APP.conversations.find(c => c.id === APP.activeId);
              if (targetConv) targetConv.working_directory = wd;
              if (UI.convWorkdirInp) UI.convWorkdirInp.value = wd;
              if (UI.convWorkdirDisplayBadge) UI.convWorkdirDisplayBadge.textContent = userPrefix + wd;
            }
          } catch (e) {
            console.error('Failed to reset conversation working directory:', e);
          }
        } else {
          APP.newChatWorkdir = '';
          if (APP.activeConv) APP.activeConv.working_directory = '';
        }
        updateHeroWorkdirDisplay();
      });
    }

    if (UI.btnConvWorkdirSave) {
      UI.btnConvWorkdirSave.addEventListener('click', async () => {
        let newPath = UI.convWorkdirInp ? UI.convWorkdirInp.value.trim() : '';
        const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
        const userPrefix = `UserData/${username}/`;
        if (newPath.startsWith(userPrefix)) newPath = newPath.slice(userPrefix.length);

        if (!APP.activeId) {
          const conv = APP.activeConv || (APP.conversations.length > 0 ? APP.conversations[0] : null);
          if (conv) {
            APP.activeId = conv.id;
            APP.activeConv = conv;
          }
        }

        if (APP.activeId) {
          try {
            const res = await apiFetch(`/api/conversations/${APP.activeId}`, {
              method: 'PATCH',
              body: { working_directory: newPath }
            });
            if (res.ok) {
              const data = await res.json();
              let wd = data.working_directory || newPath;
              if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);
              if (APP.activeConv) APP.activeConv.working_directory = wd;
              const targetConv = APP.conversations.find(c => c.id === APP.activeId);
              if (targetConv) targetConv.working_directory = wd;
            }
          } catch (e) {
            console.error('Failed to save conversation working directory:', e);
          }
        } else {
          APP.newChatWorkdir = newPath;
          if (APP.activeConv) APP.activeConv.working_directory = newPath;
        }

        updateHeroWorkdirDisplay();
        closeConvWorkdirModal();
      });
    }

    // Modal Workspace File Browser Implementation
    let currentModalPath = '';

    function openUserFileBrowserModal(subpath) {
      if (!UI.userFileBrowserDialog) return;
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      if (UI.fbUserBadge) UI.fbUserBadge.textContent = username;

      // Determine initial subpath:
      let pathToGo = '';
      if (typeof subpath === 'string') {
        pathToGo = subpath;
      } else if (APP.activeConv && APP.activeConv.working_directory) {
        pathToGo = APP.activeConv.working_directory;
      }
      const userPrefix = `UserData/${username}/`;
      if (pathToGo.startsWith(userPrefix)) pathToGo = pathToGo.slice(userPrefix.length);

      updateModalWorkdirBadge();
      UI.userFileBrowserDialog.classList.add('show');
      loadModalUserFiles(pathToGo);
    }

    function closeUserFileBrowserModal() {
      if (UI.userFileBrowserDialog) UI.userFileBrowserDialog.classList.remove('show');
      closeModalFilePreview();
      if (UI.fbNewFolderBox) UI.fbNewFolderBox.style.display = 'none';
    }

    function updateModalWorkdirBadge() {
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      const conv = APP.activeConv;
      let wd = (conv && conv.working_directory) ? conv.working_directory : '';
      const userPrefix = `UserData/${username}/`;
      if (wd.startsWith(userPrefix)) wd = wd.slice(userPrefix.length);
      const displayWd = wd ? `${userPrefix}${wd}` : `${userPrefix}(root)`;
      if (UI.fbChatWorkdirText) UI.fbChatWorkdirText.textContent = displayWd;
    }

    if (UI.btnCloseFileBrowser) UI.btnCloseFileBrowser.addEventListener('click', closeUserFileBrowserModal);
    if (UI.btnFbCloseFooter) UI.btnFbCloseFooter.addEventListener('click', closeUserFileBrowserModal);
    if (UI.userFileBrowserDialog) {
      UI.userFileBrowserDialog.addEventListener('click', (e) => {
        if (e.target === UI.userFileBrowserDialog) closeUserFileBrowserModal();
      });
    }

    async function loadModalUserFiles(subpath = '') {
      currentModalPath = subpath || '';
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      renderModalBreadcrumbs(currentModalPath, username);
      closeModalFilePreview();

      if (!UI.fbFilesListView) return;
      UI.fbFilesListView.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; padding:16px; text-align:center;">Loading files...</div>';

      try {
        const url = `/api/user-files?path=${encodeURIComponent(currentModalPath)}`;
        const res = await apiFetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const entries = data.entries || [];
        renderModalFilesList(entries, currentModalPath);
      } catch (err) {
        UI.fbFilesListView.innerHTML = `<div style="color:var(--danger); font-size:0.8rem; padding:16px; text-align:center;">Failed to load files: ${escapeHtml(err.message)}</div>`;
        if (UI.fbStatusStats) UI.fbStatusStats.textContent = 'Error loading directory';
      }
    }

    function renderModalBreadcrumbs(curPath, username) {
      if (!UI.fbBreadcrumbs) return;
      UI.fbBreadcrumbs.innerHTML = '';

      // Root crumb: UserData / <username>
      const rootCrumb = document.createElement('span');
      rootCrumb.className = 'fb-crumb' + (!curPath ? ' active' : '');
      rootCrumb.textContent = `UserData/${username}`;
      rootCrumb.onclick = () => loadModalUserFiles('');
      UI.fbBreadcrumbs.appendChild(rootCrumb);

      if (!curPath) return;

      const segments = curPath.split(/[\\/]/).filter(Boolean);
      let cumulative = '';
      segments.forEach((seg, i) => {
        const sep = document.createElement('span');
        sep.className = 'fb-crumb-sep';
        sep.textContent = '/';
        UI.fbBreadcrumbs.appendChild(sep);

        cumulative = cumulative ? `${cumulative}/${seg}` : seg;
        const crumb = document.createElement('span');
        const isLast = (i === segments.length - 1);
        crumb.className = 'fb-crumb' + (isLast ? ' active' : '');
        crumb.textContent = seg;
        const targetPath = cumulative;
        if (!isLast) {
          crumb.onclick = () => loadModalUserFiles(targetPath);
        }
        UI.fbBreadcrumbs.appendChild(crumb);
      });
    }

    function renderModalFilesList(entries, curPath) {
      if (!UI.fbFilesListView) return;
      UI.fbFilesListView.innerHTML = '';

      if (entries.length === 0) {
        UI.fbFilesListView.innerHTML = `
          <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:32px; color:var(--text-dim); gap:6px;">
            <span style="font-size:2rem;">📂</span>
            <div style="font-size:0.85rem; font-weight:500;">Empty Directory</div>
            <div style="font-size:0.75rem;">Create a folder or upload files to this workspace.</div>
          </div>
        `;
        if (UI.fbStatusStats) UI.fbStatusStats.textContent = '0 items';
        return;
      }

      // Sort: directories first, then alphabetical
      entries.sort((a, b) => {
        if (a.is_dir && !b.is_dir) return -1;
        if (!a.is_dir && b.is_dir) return 1;
        return a.name.localeCompare(b.name);
      });

      let dirCount = 0;
      let fileCount = 0;

      entries.forEach(entry => {
        if (entry.is_dir) dirCount++; else fileCount++;
        const row = document.createElement('div');
        row.className = 'fb-file-row';

        const isImg = !entry.is_dir && entry.name.match(/\\.(png|jpg|jpeg|webp|gif|svg)$/i);
        const icon = entry.is_dir ? '📁' : (isImg ? '🖼️' : '📄');
        const sizeStr = entry.is_dir ? '' : formatBytes(entry.size || 0);

        row.innerHTML = `
          <div class="fb-file-left">
            <span style="font-size:1.15rem; flex-shrink:0;">${icon}</span>
            <span class="fb-file-name ${entry.is_dir ? 'is-folder' : ''}">${escapeHtml(entry.name)}</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            ${sizeStr ? `<span style="font-size:0.72rem; color:var(--text-dim); font-family:var(--font-mono);">${sizeStr}</span>` : ''}
            ${!entry.is_dir ? `<button class="btn-top btn-fb-view" type="button" title="View / Preview" style="padding:2px 7px; font-size:0.72rem;">👁️</button>` : ''}
            ${!entry.is_dir ? `<button class="btn-top btn-fb-dl" type="button" title="Download" style="padding:2px 7px; font-size:0.72rem;">⬇️</button>` : ''}
            <button class="btn-top btn-fb-delete" type="button" title="Delete" style="padding:2px 7px; font-size:0.72rem; color:var(--danger); border-color:rgba(239,68,68,0.2);">&times;</button>
          </div>
        `;

        const fileLeft = row.querySelector('.fb-file-left');
        fileLeft.addEventListener('click', () => {
          if (entry.is_dir) {
            const nextPath = curPath ? `${curPath}/${entry.name}` : entry.name;
            loadModalUserFiles(nextPath);
          } else {
            const rel = curPath ? `${curPath}/${entry.name}` : entry.name;
            openModalFilePreview(rel, entry.name, isImg);
          }
        });

        const viewBtn = row.querySelector('.btn-fb-view');
        if (viewBtn) {
          viewBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const rel = curPath ? `${curPath}/${entry.name}` : entry.name;
            openModalFilePreview(rel, entry.name, isImg);
          });
        }

        const dlBtn = row.querySelector('.btn-fb-dl');
        if (dlBtn) {
          dlBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const rel = curPath ? `${curPath}/${entry.name}` : entry.name;
            downloadModalFile(rel, entry.name);
          });
        }

        const delBtn = row.querySelector('.btn-fb-delete');
        if (delBtn) {
          delBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const rel = curPath ? `${curPath}/${entry.name}` : entry.name;
            if (confirm(`Delete "\${entry.name}"? This cannot be undone.`)) {
              try {
                const res = await apiFetch(`/api/user-files?path=${encodeURIComponent(rel)}`, { method: 'DELETE' });
                if (res.ok) {
                  loadModalUserFiles(curPath);
                } else {
                  const d = await res.json();
                  alert('Delete failed: ' + (d.error || 'Server error'));
                }
              } catch (err) {
                alert('Delete failed: ' + err.message);
              }
            }
          });
        }

        UI.fbFilesListView.appendChild(row);
      });

      if (UI.fbStatusStats) {
        UI.fbStatusStats.textContent = `${dirCount} folder(s), ${fileCount} file(s)`;
      }
    }

    async function openModalFilePreview(relPath, fileName, isImg) {
      if (!UI.fbPreviewPane) return;
      UI.fbPreviewPane.style.display = 'flex';
      if (UI.fbPreviewName) UI.fbPreviewName.textContent = fileName || relPath;
      if (UI.fbPreviewIcon) UI.fbPreviewIcon.textContent = isImg ? '🖼️' : '📄';
      if (UI.fbPreviewText) UI.fbPreviewText.textContent = 'Loading content...';
      if (UI.fbPreviewImg) UI.fbPreviewImg.style.display = 'none';
      if (UI.fbPreviewSize) UI.fbPreviewSize.textContent = '';

      if (UI.btnFbPreviewDownload) {
        UI.btnFbPreviewDownload.onclick = () => downloadModalFile(relPath, fileName);
      }
      if (UI.btnFbPreviewCopy) {
        UI.btnFbPreviewCopy.onclick = () => {
          if (UI.fbPreviewText && UI.fbPreviewText.textContent) {
            navigator.clipboard.writeText(UI.fbPreviewText.textContent);
            UI.btnFbPreviewCopy.textContent = '✓ Copied!';
            setTimeout(() => { if (UI.btnFbPreviewCopy) UI.btnFbPreviewCopy.textContent = '📋 Copy'; }, 1500);
          }
        };
      }

      try {
        const res = await apiFetch(`/api/user-files/content?path=${encodeURIComponent(relPath)}`);
        if (!res.ok) throw new Error(`HTTP \${res.status}`);
        const data = await res.json();
        const content = data.content || '';
        if (isImg) {
          if (UI.fbPreviewText) UI.fbPreviewText.style.display = 'none';
          if (UI.fbPreviewImg) {
            UI.fbPreviewImg.src = `data:image/png;base64,\${content}`;
            UI.fbPreviewImg.style.display = 'block';
          }
        } else {
          if (UI.fbPreviewImg) UI.fbPreviewImg.style.display = 'none';
          if (UI.fbPreviewText) {
            UI.fbPreviewText.style.display = 'block';
            UI.fbPreviewText.textContent = content;
          }
        }
        if (UI.fbPreviewSize) {
          UI.fbPreviewSize.textContent = formatBytes(data.size || content.length);
        }
      } catch (err) {
        if (UI.fbPreviewText) {
          UI.fbPreviewText.style.display = 'block';
          UI.fbPreviewText.textContent = 'Error loading content: ' + err.message;
        }
      }
    }

    function closeModalFilePreview() {
      if (UI.fbPreviewPane) UI.fbPreviewPane.style.display = 'none';
    }

    if (UI.btnFbPreviewClose) UI.btnFbPreviewClose.addEventListener('click', closeModalFilePreview);

    function downloadModalFile(relPath, fileName) {
      const a = document.createElement('a');
      a.href = `/api/user-files/content?path=${encodeURIComponent(relPath)}`;
      a.download = fileName || 'download';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    // Set Currently Browsed Folder as Active Chat Working Directory
    if (UI.btnFbSetWorkdir) {
      UI.btnFbSetWorkdir.addEventListener('click', async () => {
        const newPath = currentModalPath || '';
        const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
        if (!APP.activeId) {
          const conv = APP.activeConv || (APP.conversations.length > 0 ? APP.conversations[0] : null);
          if (conv) {
            APP.activeId = conv.id;
            APP.activeConv = conv;
          }
        }
        if (APP.activeId) {
          try {
            const res = await apiFetch(`/api/conversations/${APP.activeId}`, {
              method: 'PATCH',
              body: { working_directory: newPath }
            });
            if (res.ok) {
              const data = await res.json();
              if (APP.activeConv) APP.activeConv.working_directory = data.working_directory || newPath;
              const targetConv = APP.conversations.find(c => c.id === APP.activeId);
              if (targetConv) targetConv.working_directory = data.working_directory || newPath;
              updateHeroWorkdirDisplay();
              updateModalWorkdirBadge();
              alert(`Chat working directory updated:\nUserData/${username}/${newPath ? newPath : '(root)'}`);
            }
          } catch (e) {
            alert('Failed to set chat working directory: ' + e.message);
          }
        } else {
          APP.newChatWorkdir = newPath;
          if (APP.activeConv) APP.activeConv.working_directory = newPath;
          updateHeroWorkdirDisplay();
          updateModalWorkdirBadge();
          alert(`Chat working directory updated:\nUserData/${username}/${newPath ? newPath : '(root)'}`);
        }
      });
    }

    // Modal Folder Creation & File Upload
    if (UI.btnFbNewFolder) {
      UI.btnFbNewFolder.addEventListener('click', () => {
        if (UI.fbNewFolderBox) {
          const isOpen = UI.fbNewFolderBox.style.display === 'flex';
          UI.fbNewFolderBox.style.display = isOpen ? 'none' : 'flex';
          if (!isOpen && UI.fbNewFolderInp) {
            UI.fbNewFolderInp.value = '';
            UI.fbNewFolderInp.focus();
          }
        }
      });
    }
    if (UI.btnFbCancelFolder) {
      UI.btnFbCancelFolder.addEventListener('click', () => {
        if (UI.fbNewFolderBox) UI.fbNewFolderBox.style.display = 'none';
      });
    }
    if (UI.btnFbCreateFolderExec) {
      UI.btnFbCreateFolderExec.addEventListener('click', async () => {
        const name = UI.fbNewFolderInp ? UI.fbNewFolderInp.value.trim() : '';
        if (!name) return;
        try {
          const res = await apiFetch('/api/user-folders', {
            method: 'POST',
            body: { name: name, parent: currentModalPath }
          });
          if (res.ok) {
            if (UI.fbNewFolderBox) UI.fbNewFolderBox.style.display = 'none';
            loadModalUserFiles(currentModalPath);
          } else {
            const d = await res.json();
            alert('Failed to create folder: ' + (d.error || 'Server error'));
          }
        } catch (err) {
          alert('Error creating folder: ' + err.message);
        }
      });
    }

    if (UI.btnFbUpload) {
      UI.btnFbUpload.addEventListener('click', () => {
        if (UI.fbFileUploadInput) UI.fbFileUploadInput.click();
      });
    }
    if (UI.fbFileUploadInput) {
      UI.fbFileUploadInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length === 0) return;
        for (const file of files) {
          try {
            const content = await readFileAsText(file);
            const targetRelPath = currentModalPath ? `${currentModalPath}/${file.name}` : file.name;
            await apiFetch('/api/user-files/content', {
              method: 'POST',
              body: { path: targetRelPath, content: content }
            });
          } catch (err) {
            alert(`Failed to upload ${file.name}: ${err.message}`);
          }
        }
        UI.fbFileUploadInput.value = '';
        loadModalUserFiles(currentModalPath);
      });
    }

    if (UI.btnFbUp) {
      UI.btnFbUp.addEventListener('click', () => {
        if (!currentModalPath) return;
        const parts = currentModalPath.split(/[\\/]/).filter(Boolean);
        parts.pop();
        loadModalUserFiles(parts.join('/'));
      });
    }
    if (UI.btnFbRefresh) {
      UI.btnFbRefresh.addEventListener('click', () => loadModalUserFiles(currentModalPath));
    }

    // Tools & MCP Management
    function setMasterTools(enabled) {
      APP.toolsEnabled = !!enabled;
      localStorage.setItem('gufo_tools_enabled', APP.toolsEnabled ? 'true' : 'false');
      if (UI.chkMasterTools) UI.chkMasterTools.checked = APP.toolsEnabled;
      if (UI.modalChkMasterTools) UI.modalChkMasterTools.checked = APP.toolsEnabled;
      updateToolsPill();
      saveInbuiltTools();
    }

    if (UI.chkMasterTools) {
      UI.chkMasterTools.addEventListener('change', (e) => {
        setMasterTools(e.target.checked);
      });
    }
    if (UI.modalChkMasterTools) {
      UI.modalChkMasterTools.addEventListener('change', (e) => {
        setMasterTools(e.target.checked);
      });
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
          if (data.server_current_directory) {
            APP.serverCurrentDirectory = data.server_current_directory;
          }
          if (data.server_tools_override) {
            APP.toolsEnabled = !!data.master_enabled;
            localStorage.setItem('gufo_tools_enabled', APP.toolsEnabled ? '1' : '0');
          } else if (typeof data.master_enabled === 'boolean') {
            if (localStorage.getItem('gufo_tools_enabled') === null) {
              APP.toolsEnabled = data.master_enabled;
            }
          }
          renderInbuiltTools();
          renderMcpServers();
          renderCustomTools();
          updateToolsPill();
          updateAttachPopupUI();
        }
      } catch (err) {
        console.error('Failed to load tools:', err);
      }
    }

    function updateToolsPill() {
      const activeInbuilt = (APP.tools.inbuilt || []).filter(t => t.name !== 'ask_user' && t.enabled).length;
      const activeCustom = (APP.tools.customTools || []).filter(t => t.enabled).length;
      const total = activeInbuilt + activeCustom;

      if (UI.attachToolsBadge) {
        UI.attachToolsBadge.classList.toggle('disabled', !APP.toolsEnabled);
      }
      if (UI.attachToolsStatusText) {
        UI.attachToolsStatusText.textContent = APP.toolsEnabled ? `${total} tools active` : 'Tools disabled';
      }
      if (UI.navToolsLabel) {
        UI.navToolsLabel.textContent = APP.toolsEnabled ? `Tools & MCP (${total})` : 'Tools: OFF';
      }
      if (UI.chkMasterTools) {
        UI.chkMasterTools.checked = APP.toolsEnabled;
      }
      if (UI.modalChkMasterTools) {
        UI.modalChkMasterTools.checked = APP.toolsEnabled;
      }
    }

    function renderInbuiltTools() {
      if (!UI.inbuiltToolsList) return;
      UI.inbuiltToolsList.innerHTML = '';
      const visibleTools = (APP.tools.inbuilt || []).filter(t => t.name !== 'ask_user');
      if (visibleTools.length === 0) {
        UI.inbuiltToolsList.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; padding:8px;">No inbuilt tools available.</div>';
        return;
      }
      visibleTools.forEach(t => {
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
          body: {
            master_enabled: APP.toolsEnabled,
            enabled: enabled
          }
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
        if (c.is_generating) {
          const genDot = document.createElement('span');
          genDot.style.cssText = 'color:var(--accent); font-size:0.8rem; margin-right:4px; display:inline-block;';
          genDot.title = 'Generating response...';
          genDot.textContent = '⚡ ';
          title.prepend(genDot);
        }

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
          if (window.innerWidth <= 768) toggleSidebar(false);
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
      if (window.innerWidth <= 768) toggleSidebar(false);
      APP.activeId = id;
      renderChatList();
      try {
        const res = await apiFetch(`/api/conversations/${id}`);
        if (res.ok) {
          APP.activeConv = await res.json();
          renderMessages(APP.activeConv.messages || []);
          updateAttachPopupUI();
          updateHeroWorkdirDisplay();
          updateInputMetaBar();
          if (APP.activeConv.is_generating) {
            resumeLiveStream(id, !!APP.activeConv.is_resume);
          }
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
          const data = await res.json();
          const item = APP.conversations.find(c => c.id === id);
          if (item) {
            item.title = title;
            if (data.working_directory) item.working_directory = data.working_directory;
          }
          if (APP.activeConv && APP.activeConv.id === id) {
            APP.activeConv.title = title;
            if (data.working_directory) APP.activeConv.working_directory = data.working_directory;
            updateAttachPopupUI();
          }
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

    function saveMessageBadge(convId, m, idx, badgeHtml) {
      if (!convId || !badgeHtml) return;
      try {
        const storageKey = 'gufo_msg_badges_' + convId;
        const badges = JSON.parse(localStorage.getItem(storageKey) || '{}');
        if (idx != null) badges['idx_' + idx] = badgeHtml;
        if (m && m.id) badges['id_' + m.id] = badgeHtml;
        if (m && m.content) {
          badges['c_' + m.content.slice(0, 60).replace(/\s+/g, ' ')] = badgeHtml;
        }
        localStorage.setItem(storageKey, JSON.stringify(badges));
      } catch (_) {}
    }

    function getMessageBadge(convId, m, idx) {
      if (!convId) return null;
      try {
        const storageKey = 'gufo_msg_badges_' + convId;
        const badges = JSON.parse(localStorage.getItem(storageKey) || '{}');
        if (m && m.id && badges['id_' + m.id]) return badges['id_' + m.id];
        if (m && m.content) {
          const cKey = 'c_' + m.content.slice(0, 60).replace(/\s+/g, ' ');
          if (badges[cKey]) return badges[cKey];
        }
        if (idx != null && badges['idx_' + idx]) return badges['idx_' + idx];
        return null;
      } catch (_) {
        return null;
      }
    }

    // Render Messages
    function renderMessages(messages) {
      UI.messagesContainer.innerHTML = '';
      if (!messages || messages.length === 0) {
        renderHeroDashboard();
        return;
      }

      messages.forEach((m, idx) => appendMessage(m, idx));
      isUserScrolledUp = false;
      if (UI.btnScrollBottom) UI.btnScrollBottom.classList.remove('visible');
      scrollDown(true);
    }

    function renderHeroDashboard() {
      const username = (APP.user && APP.user.username) ? APP.user.username : 'Guest';
      const userPrefix = `UserData/${username}/`;
      let rawWd = '';
      if (APP.activeConv && APP.activeConv.working_directory) rawWd = APP.activeConv.working_directory;
      else if (APP.newChatWorkdir) rawWd = APP.newChatWorkdir;
      if (rawWd.startsWith(userPrefix)) rawWd = rawWd.slice(userPrefix.length);
      const wdName = rawWd ? `UserData/${username}/${rawWd}` : `UserData/${username}/...`;

      const activeInbuilt = (APP.tools.inbuilt || []).filter(t => t.name !== 'ask_user' && t.enabled).length;
      const activeCustom = (APP.tools.customTools || []).filter(t => t.enabled).length;
      const totalTools = activeInbuilt + activeCustom;
      const toolsDesc = APP.toolsEnabled ? `${totalTools} tools active` : 'Tools disabled';
      const permDesc = APP.toolPermission === 'allow' ? 'Allow all (autonomous)' : 'Ask before running';
      const thinkDesc = `${APP.thinkingLevel} reasoning`;

      const activePromptMeta = getActivePromptMeta();
      const activePromptKey = getActivePromptKey();

      let renderedPromptThumbs = PROMPT_PRESETS.map(p => `
        <div class="hero-prompt-thumb ${p.key === activePromptKey ? 'active' : ''}" data-key="${p.key}" onclick="event.stopPropagation(); selectSystemPrompt('${p.key}')" title="${escapeHtml(p.desc)}">
          <div class="hero-prompt-thumb-icon">${p.icon}</div>
          <div class="hero-prompt-thumb-info">
            <div class="hero-prompt-thumb-title">${escapeHtml(p.name)}</div>
            <div class="hero-prompt-thumb-sub">${escapeHtml(p.tag)}</div>
          </div>
        </div>
      `).join('');

      if (APP.customPrompts && APP.customPrompts.length > 0) {
        renderedPromptThumbs += APP.customPrompts.map((cp, idx) => {
          const k = `custom_${idx}`;
          return `
            <div class="hero-prompt-thumb ${k === activePromptKey ? 'active' : ''}" data-key="${k}" onclick="event.stopPropagation(); selectSystemPrompt('${k}')" title="${escapeHtml(cp.prompt)}">
              <div class="hero-prompt-thumb-icon">⭐</div>
              <div class="hero-prompt-thumb-info">
                <div class="hero-prompt-thumb-title">${escapeHtml(cp.name)}</div>
                <div class="hero-prompt-thumb-sub">Custom preset</div>
              </div>
            </div>
          `;
        }).join('');
      }

      UI.messagesContainer.innerHTML = `
        <div class="llama-hero">
          <div class="llama-hero-icon">&#x1F989;</div>
          <div class="llama-hero-title">Gufo Chat</div>
          <div class="llama-hero-subtitle">High-performance LLM inference running natively on AMD Strix Halo gfx1151. Multi-device chat history synced to SQLite database.</div>
          <div class="hero-attach-grid">
            <!-- 1. System Prompt & Persona selection with thumbnail icons -->
            <div class="hero-attach-card" style="grid-column: 1 / -1; cursor: default;">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">🎭</span>
                  <span class="hero-attach-card-title">System Prompt &amp; Persona</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                  <span id="hero-prompt-badge" class="hero-card-badge">${activePromptMeta.icon} ${escapeHtml(activePromptMeta.name)}</span>
                  <button type="button" class="btn-mini-action" onclick="openNewPromptModal()" title="Add custom system prompt persona">+ New</button>
                </div>
              </div>
              <div class="hero-attach-card-sub">Choose AI persona or behavior instructions for this conversation:</div>
              <div class="hero-prompt-grid">
                ${renderedPromptThumbs}
              </div>
            </div>

            <!-- 2. Attach Images -->
            <div class="hero-attach-card" onclick="if(UI.fileUploaderImage) UI.fileUploaderImage.click()">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">🖼️</span>
                  <span class="hero-attach-card-title">Attach Images</span>
                </div>
                <span class="tag-pill">Vision</span>
              </div>
              <div class="hero-attach-card-sub">Add multimodal visual input to your conversation context</div>
            </div>

            <!-- 3. Attach Files -->
            <div class="hero-attach-card" onclick="if(UI.fileUploaderDoc) UI.fileUploaderDoc.click()">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">📄</span>
                  <span class="hero-attach-card-title">Attach Files</span>
                </div>
                <span class="tag-pill">Context</span>
              </div>
              <div class="hero-attach-card-sub">Attach code, logs, documents, or data files directly to prompt</div>
            </div>

            <!-- 4. Working Directory -->
            <div class="hero-attach-card" onclick="openConvWorkdirModal()">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">📁</span>
                  <span class="hero-attach-card-title">Working Directory</span>
                </div>
                <span class="tag-pill">Per-Chat</span>
              </div>
              <div id="hero-workdir-sub" class="hero-attach-card-sub">${escapeHtml(wdName)}</div>
            </div>

            <!-- 5. Tools & MCP -->
            <div class="hero-attach-card" onclick="openDrawer('tools')">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">⚙️</span>
                  <span class="hero-attach-card-title">Tools &amp; MCP</span>
                </div>
                <span class="hero-card-badge">${escapeHtml(toolsDesc)}</span>
              </div>
              <div class="hero-attach-card-sub">Filesystem, shell, code search, web, and external MCP servers</div>
            </div>

            <!-- 6. Tool Permission -->
            <div class="hero-attach-card" onclick="cycleToolPermission()">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">🛡️</span>
                  <span class="hero-attach-card-title">Tool Permission</span>
                </div>
                <span id="hero-perm-badge" class="hero-card-badge">${escapeHtml(permDesc)}</span>
              </div>
              <div class="hero-attach-card-sub">Choose between human approval ("Ask") and autonomous execution ("Allow all")</div>
              <div class="segmented-control" style="grid-template-columns:1fr 1fr; margin-top:6px;" onclick="event.stopPropagation()">
                <button type="button" class="seg-btn hero-perm-btn ${APP.toolPermission === 'ask' ? 'active' : ''}" data-perm="ask" onclick="setToolPermission('ask')" title="Prompt for approval before running tools">🛡️ Ask</button>
                <button type="button" class="seg-btn hero-perm-btn ${APP.toolPermission === 'allow' ? 'active' : ''}" data-perm="allow" onclick="setToolPermission('allow')" title="Allow autonomous tool execution without asking">⚡ Allow all</button>
              </div>
            </div>

            <!-- 7. Thinking Level -->
            <div class="hero-attach-card" onclick="cycleThinkingLevel()">
              <div class="hero-attach-card-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="hero-attach-card-icon">🧠</span>
                  <span class="hero-attach-card-title">Thinking Level</span>
                </div>
                <span id="hero-think-badge" class="hero-card-badge">${escapeHtml(thinkDesc)}</span>
              </div>
              <div class="hero-attach-card-sub">Configure reasoning depth: auto, minimum, low, medium, high, xhigh</div>
              <div class="segmented-control" style="grid-template-columns:repeat(6, 1fr); margin-top:6px;" onclick="event.stopPropagation()">
                <button type="button" class="seg-btn hero-think-btn ${APP.thinkingLevel === 'auto' ? 'active' : ''}" data-level="auto" onclick="setThinkingLevel('auto')" title="Default thinking budget">Auto</button>
                <button type="button" class="seg-btn hero-think-btn ${APP.thinkingLevel === 'minimum' ? 'active' : ''}" data-level="minimum" onclick="setThinkingLevel('minimum')" title="Fast/concise reasoning (512 tokens)">Min</button>
                <button type="button" class="seg-btn hero-think-btn ${APP.thinkingLevel === 'low' ? 'active' : ''}" data-level="low" onclick="setThinkingLevel('low')" title="Low reasoning budget (2,048 tokens)">Low</button>
                <button type="button" class="seg-btn hero-think-btn ${APP.thinkingLevel === 'medium' ? 'active' : ''}" data-level="medium" onclick="setThinkingLevel('medium')" title="Medium reasoning budget (4,096 tokens)">Med</button>
                <button type="button" class="seg-btn hero-think-btn ${APP.thinkingLevel === 'high' ? 'active' : ''}" data-level="high" onclick="setThinkingLevel('high')" title="High reasoning budget (8,192 tokens)">High</button>
                <button type="button" class="seg-btn hero-think-btn ${APP.thinkingLevel === 'xhigh' ? 'active' : ''}" data-level="xhigh" onclick="setThinkingLevel('xhigh')" title="Maximum reasoning budget (16,384 tokens)">XHigh</button>
              </div>
            </div>
          </div>
        </div>
      `;
    }

    function findMessageIndex(m, fallbackIdx) {
      if (APP.activeConv && Array.isArray(APP.activeConv.messages)) {
        if (typeof fallbackIdx === 'number' && fallbackIdx >= 0 && fallbackIdx < APP.activeConv.messages.length) {
          if (APP.activeConv.messages[fallbackIdx] === m || APP.activeConv.messages[fallbackIdx].content === m.content) {
            return fallbackIdx;
          }
        }
        const idx = APP.activeConv.messages.indexOf(m);
        if (idx !== -1) return idx;
        for (let i = APP.activeConv.messages.length - 1; i >= 0; --i) {
          if (APP.activeConv.messages[i].role === m.role && APP.activeConv.messages[i].content === m.content) {
            return i;
          }
        }
      }
      return typeof fallbackIdx === 'number' ? fallbackIdx : -1;
    }

    async function editAndResubmitMessage(targetIdx, newText) {
      if (APP.isGenerating && APP.abortController) {
        APP.abortController.abort();
        APP.isGenerating = false;
      }
      if (!APP.activeId || !APP.activeConv || !APP.activeConv.messages) return;
      if (targetIdx < 0 || targetIdx >= APP.activeConv.messages.length) return;

      // Keep messages up to targetIdx and update content
      APP.activeConv.messages = APP.activeConv.messages.slice(0, targetIdx + 1);
      APP.activeConv.messages[targetIdx].content = newText;

      try {
        const res = await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
          method: 'PUT',
          body: { messages: APP.activeConv.messages }
        });
        if (res.ok) {
          const convRes = await apiFetch(`/api/conversations/${APP.activeId}`);
          if (convRes.ok) {
            APP.activeConv = await convRes.json();
          }
        }
      } catch (err) {
        console.error('Failed to update messages on server:', err);
      }

      renderMessages(APP.activeConv.messages || []);
      await triggerAssistantTurn();
    }

    async function retryUserMessage(targetIdx) {
      if (APP.isGenerating && APP.abortController) {
        APP.abortController.abort();
        APP.isGenerating = false;
      }
      if (!APP.activeId || !APP.activeConv || !APP.activeConv.messages) return;
      if (targetIdx < 0 || targetIdx >= APP.activeConv.messages.length) return;

      // Keep messages up to targetIdx
      APP.activeConv.messages = APP.activeConv.messages.slice(0, targetIdx + 1);

      try {
        const res = await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
          method: 'PUT',
          body: { messages: APP.activeConv.messages }
        });
        if (res.ok) {
          const convRes = await apiFetch(`/api/conversations/${APP.activeId}`);
          if (convRes.ok) {
            APP.activeConv = await convRes.json();
          }
        }
      } catch (err) {
        console.error('Failed to update messages on server:', err);
      }

      renderMessages(APP.activeConv.messages || []);
      await triggerAssistantTurn();
    }

    function appendMessage(m, msgIdx) {
      if (m.role === 'tool') {
        const card = document.createElement('div');
        card.className = 'tool-call-card';
        card.innerHTML = `
          <div class="tool-call-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
            <span class="tool-call-name">🔧 Tool Result: <b>${escapeHtml(m.name || 'tool')}</b></span>
            <span class="tool-status-pill tool-status-success">✓ completed</span>
          </div>
          <div class="tool-output-box collapsed">
            <pre style="margin:0; color:var(--text); white-space:pre-wrap; word-break:break-word; overflow-wrap:break-word;">${escapeHtml(m.content || '')}</pre>
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
              <pre style="margin:0; color:var(--text); white-space:pre-wrap; word-break:break-word; overflow-wrap:break-word;">${escapeHtml(typeof fn.arguments === 'object' ? JSON.stringify(fn.arguments, null, 2) : (fn.arguments || ''))}</pre>
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

      if (isUser) {
        const btnEdit = document.createElement('button');
        btnEdit.className = 'btn-msg-action';
        btnEdit.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg> Edit';
        btnEdit.title = 'Edit message';
        btnEdit.onclick = () => {
          if (body.querySelector('.msg-edit-box')) return;
          const origContent = m.content || '';
          content.style.display = 'none';
          actions.style.display = 'none';

          const editBox = document.createElement('div');
          editBox.className = 'msg-edit-box';

          const textarea = document.createElement('textarea');
          textarea.className = 'msg-edit-textarea';
          textarea.value = origContent;

          const btnRow = document.createElement('div');
          btnRow.style.cssText = 'display:flex; justify-content:flex-end; gap:8px;';

          const btnCancel = document.createElement('button');
          btnCancel.className = 'btn-secondary';
          btnCancel.textContent = 'Cancel';
          btnCancel.style.cssText = 'padding:4px 12px; font-size:0.8rem; border-radius:var(--radius-sm); border:1px solid var(--border); background:var(--bg-card); color:var(--text-muted); cursor:pointer;';
          btnCancel.onclick = () => {
            editBox.remove();
            content.style.display = '';
            actions.style.display = 'flex';
          };

          const btnSave = document.createElement('button');
          btnSave.className = 'btn-primary';
          btnSave.textContent = 'Save & Submit';
          btnSave.style.cssText = 'padding:4px 12px; font-size:0.8rem; border-radius:var(--radius-sm); border:none; background:var(--primary); color:#fff; font-weight:500; cursor:pointer;';
          btnSave.onclick = () => {
            const updated = textarea.value.trim();
            if (!updated) return;
            const target = findMessageIndex(m, msgIdx);
            editAndResubmitMessage(target, updated);
          };

          textarea.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
              e.preventDefault();
              btnSave.click();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              btnCancel.click();
            }
          });

          btnRow.appendChild(btnCancel);
          btnRow.appendChild(btnSave);
          editBox.appendChild(textarea);
          editBox.appendChild(btnRow);
          body.appendChild(editBox);

          textarea.focus();
          textarea.selectionStart = textarea.selectionEnd = textarea.value.length;
        };
        actions.appendChild(btnEdit);

        const btnRetry = document.createElement('button');
        btnRetry.className = 'btn-msg-action';
        btnRetry.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg> Regnerate';
        btnRetry.title = 'Regnerate message';
        btnRetry.onclick = () => {
          const target = findMessageIndex(m, msgIdx);
          retryUserMessage(target);
        };
        actions.appendChild(btnRetry);
      }

      const btnCopy = document.createElement('button');
      btnCopy.className = 'btn-msg-action';
      btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
      btnCopy.title = 'Copy message';
      btnCopy.onclick = () => {
        navigator.clipboard.writeText(m.content || '');
        btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
        setTimeout(() => {
          btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
        }, 1500);
      };
      actions.appendChild(btnCopy);

      if (!isUser) {
        const savedBadgeHtml = m.usage_badge || getMessageBadge(APP.activeId, m, msgIdx);
        if (savedBadgeHtml) {
          const usageBadge = document.createElement('div');
          usageBadge.className = 'msg-usage-badge';
          usageBadge.innerHTML = savedBadgeHtml;
          actions.appendChild(usageBadge);
        }
      }

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

    let isUserScrolledUp = false;
    let programmaticScroll = false;

    if (UI.messagesContainer) {
      // Detect manual user intent to scroll up (wheel or touch gesture)
      UI.messagesContainer.addEventListener('wheel', (e) => {
        if (e.deltaY < 0) {
          isUserScrolledUp = true;
          if (UI.btnScrollBottom) UI.btnScrollBottom.classList.add('visible');
        } else if (e.deltaY > 0) {
          const distFromBottom = UI.messagesContainer.scrollHeight - UI.messagesContainer.scrollTop - UI.messagesContainer.clientHeight;
          if (distFromBottom <= 100) {
            isUserScrolledUp = false;
            if (UI.btnScrollBottom) UI.btnScrollBottom.classList.remove('visible');
          }
        }
      }, { passive: true });

      let touchStartY = 0;
      UI.messagesContainer.addEventListener('touchstart', (e) => {
        if (e.touches && e.touches[0]) touchStartY = e.touches[0].clientY;
      }, { passive: true });

      UI.messagesContainer.addEventListener('touchmove', (e) => {
        if (e.touches && e.touches[0]) {
          const deltaY = e.touches[0].clientY - touchStartY;
          if (deltaY > 10) {
            isUserScrolledUp = true;
            if (UI.btnScrollBottom) UI.btnScrollBottom.classList.add('visible');
          } else if (deltaY < -10) {
            const distFromBottom = UI.messagesContainer.scrollHeight - UI.messagesContainer.scrollTop - UI.messagesContainer.clientHeight;
            if (distFromBottom <= 100) {
              isUserScrolledUp = false;
              if (UI.btnScrollBottom) UI.btnScrollBottom.classList.remove('visible');
            }
          }
        }
      }, { passive: true });

      UI.messagesContainer.addEventListener('scroll', () => {
        if (programmaticScroll) {
          programmaticScroll = false;
          return;
        }
        const distFromBottom = UI.messagesContainer.scrollHeight - UI.messagesContainer.scrollTop - UI.messagesContainer.clientHeight;
        if (distFromBottom <= 80) {
          isUserScrolledUp = false;
        } else if (distFromBottom > 160) {
          isUserScrolledUp = true;
        }
        if (UI.btnScrollBottom) {
          UI.btnScrollBottom.classList.toggle('visible', isUserScrolledUp);
        }
      });
    }

    if (UI.btnScrollBottom) {
      UI.btnScrollBottom.addEventListener('click', () => {
        isUserScrolledUp = false;
        if (UI.btnScrollBottom) UI.btnScrollBottom.classList.remove('visible');
        if (UI.messagesContainer) {
          programmaticScroll = true;
          UI.messagesContainer.scrollTo({ top: UI.messagesContainer.scrollHeight, behavior: 'smooth' });
        }
      });
    }

    function scrollDown(force = false) {
      if (!UI.messagesContainer) return;
      if (force || !isUserScrolledUp) {
        programmaticScroll = true;
        UI.messagesContainer.scrollTop = UI.messagesContainer.scrollHeight;
      }
    }

    function updateLiveSpeed(genTokens, startTime) {
      if (!startTime || genTokens <= 0) return;
      const elapsedSec = (performance.now() - startTime) / 1000;
      if (elapsedSec > 0.1) {
        const tps = (genTokens / elapsedSec).toFixed(1);
        if (UI.footerSpeedMetrics) {
          UI.footerSpeedMetrics.textContent = `eval: ${tps} t/s | ${genTokens} tokens`;
        }
        if (UI.slotText) {
          UI.slotText.textContent = `1/1 slots (generating: ${tps} t/s)`;
        }
        APP.lastSeenGenSpeed = Number(tps);
        APP.lastSeenGenTokens = genTokens;
      }
    }

    // Image upload & paste handling
    if (UI.fileUploaderImage) {
      UI.fileUploaderImage.addEventListener('change', e => {
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
        if (UI.fileUploaderImage) UI.fileUploaderImage.value = '';
      });
    }

    // Document & Code Files Upload Handling
    if (UI.fileUploaderDoc) {
      UI.fileUploaderDoc.addEventListener('change', async e => {
        const files = Array.from(e.target.files || []);
        for (const file of files) {
          try {
            const content = await readFileAsText(file);
            APP.attachedFiles.push({
              name: file.name,
              size: file.size,
              content: content
            });
          } catch (err) {
            console.error('Failed to read file:', file.name, err);
          }
        }
        UI.fileUploaderDoc.value = '';
        renderAttachedFilesBar();
        calculateContextSize();
      });
    }

    function readFileAsText(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
      });
    }

    function renderAttachedFilesBar() {
      if (!UI.attachedFilesBar) return;
      if (!APP.attachedFiles || APP.attachedFiles.length === 0) {
        UI.attachedFilesBar.style.display = 'none';
        UI.attachedFilesBar.innerHTML = '';
        return;
      }
      UI.attachedFilesBar.style.display = 'flex';
      UI.attachedFilesBar.innerHTML = '';
      APP.attachedFiles.forEach((f, idx) => {
        const pill = document.createElement('div');
        pill.className = 'attached-file-pill';
        const formattedSize = f.size > 1024 ? `${(f.size / 1024).toFixed(1)} KB` : `${f.size} B`;
        pill.innerHTML = `
          <span>📄</span>
          <span style="font-weight:500; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(f.name)}</span>
          <span class="file-size">(${formattedSize})</span>
          <button type="button" class="btn-remove-file" title="Remove attachment">&times;</button>
        `;
        pill.querySelector('.btn-remove-file').addEventListener('click', (e) => {
          e.stopPropagation();
          APP.attachedFiles.splice(idx, 1);
          renderAttachedFilesBar();
          calculateContextSize();
        });
        UI.attachedFilesBar.appendChild(pill);
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
        if (APP.activeId) {
          apiFetch(`/api/conversations/${APP.activeId}/stop`, { method: 'POST' }).catch(() => {});
        }
        if (APP.abortController) APP.abortController.abort();
      });
    }

    const ASK_USER_TOOL = {
      type: 'function',
      function: {
        name: 'ask_user',
        description: 'Ask the user a question with optional choice buttons or text write-in when you need more information, clarification, or confirmation.',
        parameters: {
          type: 'object',
          properties: {
            question: {
              type: 'string',
              description: 'The question or prompt to present to the user.'
            },
            options: {
              type: 'array',
              items: { type: 'string' },
              description: 'Optional list of quick selectable choices for the user (e.g. ["Yes", "No"]).'
            },
            allow_text_input: {
              type: 'boolean',
              description: 'Whether the user can provide a freeform text answer (default true).'
            }
          },
          required: ['question']
        }
      }
    };

    function getActiveToolsList() {
      // ask_user tool always stays active without needing to be configured in settings
      const tools = [{ ...ASK_USER_TOOL }];
      if (!APP.toolsEnabled) return tools;
      if (APP.tools.inbuilt) {
        for (const t of APP.tools.inbuilt) {
          if (t.name === 'ask_user') continue;
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

    // Interactive Model Clarification Question
    function askUserInquiry(tc, parsedArgs) {
      return new Promise((resolve) => {
        const card = document.createElement('div');
        card.className = 'tool-call-card inquiry-card';
        card.style.borderColor = 'var(--accent)';
        card.style.background = 'rgba(59, 130, 246, 0.08)';

        const qText = parsedArgs.question || parsedArgs.q || 'Gufo is asking for your input:';
        const options = Array.isArray(parsedArgs.options) ? parsedArgs.options : [];
        const allowText = parsedArgs.allow_text_input !== false;

        let optionsHtml = '';
        if (options.length > 0) {
          optionsHtml = `<div style="display:flex; flex-wrap:wrap; gap:8px; margin:10px 0;">` +
            options.map((opt, i) => `<button type="button" class="inquiry-opt-btn" data-idx="${i}">${escapeHtml(opt)}</button>`).join('') +
            `</div>`;
        }

        let inputHtml = '';
        if (allowText) {
          inputHtml = `
            <div style="display:flex; gap:8px; margin-top:8px;">
              <input type="text" class="popup-select inquiry-text-input" placeholder="Type your answer..." style="flex:1;">
              <button type="button" class="btn-top inquiry-submit-btn" style="background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; padding:6px 14px;">Send Answer</button>
            </div>
          `;
        }

        card.innerHTML = `
          <div class="tool-call-header" style="cursor:default;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-size:1.1rem;">❓</span>
              <span class="tool-call-name" style="color:var(--text);">Model Clarification Question</span>
            </div>
            <span class="tool-status-pill tool-status-running" style="background:rgba(59, 130, 246, 0.2); color:#60a5fa; border:1px solid rgba(59, 130, 246, 0.4);">Awaiting User Response</span>
          </div>
          <div style="padding:10px 14px;">
            <div style="font-size:0.92rem; color:var(--text); font-weight:500; margin-bottom:4px; line-height:1.5;">${escapeHtml(qText)}</div>
            ${optionsHtml}
            ${inputHtml}
          </div>
        `;

        UI.messagesContainer.appendChild(card);
        scrollDown();

        const finishInquiry = (answer) => {
          card.querySelectorAll('button, input').forEach(el => el.disabled = true);
          const statusPill = card.querySelector('.tool-status-pill');
          if (statusPill) {
            statusPill.className = 'tool-status-pill tool-status-success';
            statusPill.textContent = '✓ Answered';
          }
          const answerBadge = document.createElement('div');
          answerBadge.style.margin = '8px 14px 10px 14px';
          answerBadge.style.padding = '6px 10px';
          answerBadge.style.borderRadius = 'var(--radius-sm)';
          answerBadge.style.background = 'rgba(34, 197, 94, 0.1)';
          answerBadge.style.border = '1px solid rgba(34, 197, 94, 0.3)';
          answerBadge.style.fontSize = '0.82rem';
          answerBadge.style.color = '#4ade80';
          answerBadge.innerHTML = `✓ You answered: <b>${escapeHtml(answer)}</b>`;
          card.appendChild(answerBadge);
          scrollDown();
          resolve(answer);
        };

        const optBtns = card.querySelectorAll('.inquiry-opt-btn');
        optBtns.forEach(btn => {
          btn.addEventListener('click', () => finishInquiry(btn.textContent.trim()));
        });

        const submitBtn = card.querySelector('.inquiry-submit-btn');
        const textInp = card.querySelector('.inquiry-text-input');
        if (submitBtn && textInp) {
          submitBtn.addEventListener('click', () => {
            const val = textInp.value.trim();
            if (val) finishInquiry(val);
          });
          textInp.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
              const val = textInp.value.trim();
              if (val) finishInquiry(val);
            }
          });
          textInp.focus();
        }
      });
    }

    // Human Approval for Tool Execution (Ask before running)
    function askToolApproval(tc, parsedArgs) {
      return new Promise((resolve) => {
        const card = document.createElement('div');
        card.className = 'tool-call-card';
        card.style.borderColor = 'rgba(234, 179, 8, 0.5)';
        card.style.background = 'rgba(234, 179, 8, 0.05)';

        card.innerHTML = `
          <div class="tool-call-header" style="cursor:default;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-size:1.1rem;">🛡️</span>
              <span class="tool-call-name">Tool Permission Request: <b>${escapeHtml(tc.name)}</b></span>
            </div>
            <span class="tool-status-pill" style="background:rgba(234, 179, 8, 0.2); color:#facc15; border:1px solid rgba(234, 179, 8, 0.4);">Permission Required</span>
          </div>
          <div style="padding:10px 14px;">
            <div style="color:var(--text-dim); margin-bottom:4px; font-size:0.75rem;">Arguments requested by model:</div>
            <pre style="margin:0 0 10px 0; max-height:160px; overflow-y:auto; background:rgba(0,0,0,0.4); padding:8px; border-radius:var(--radius-sm); font-size:0.75rem; color:var(--text); font-family:var(--font-mono);">${escapeHtml(JSON.stringify(parsedArgs, null, 2))}</pre>
            <div style="display:flex; gap:8px;">
              <button type="button" class="btn-top btn-approve-run" style="background:#22c55e; color:#fff; border-color:#22c55e; font-weight:600; padding:6px 14px;">✓ Approve & Run</button>
              <button type="button" class="btn-top btn-deny-run" style="background:transparent; color:var(--danger); border-color:var(--danger); font-weight:600; padding:6px 14px;">✕ Deny</button>
            </div>
          </div>
        `;
        UI.messagesContainer.appendChild(card);
        scrollDown();

        const btnApprove = card.querySelector('.btn-approve-run');
        const btnDeny = card.querySelector('.btn-deny-run');

        btnApprove.addEventListener('click', () => {
          card.remove();
          resolve(true);
        });

        btnDeny.addEventListener('click', () => {
          card.innerHTML = `
            <div style="padding:8px 12px; display:flex; align-items:center; gap:6px; color:var(--danger); font-size:0.8rem;">
              <span>✕</span> Tool <b>${escapeHtml(tc.name)}</b> execution denied by user.
            </div>
          `;
          resolve(false);
        });
      });
    }

    // Send Message with Live SSE Streaming & Autonomous Tool Execution
    async function sendMessage() {
      const rawText = UI.chatInput.value.trim();
      const hasFiles = APP.attachedFiles && APP.attachedFiles.length > 0;
      if ((!rawText && !APP.attachedImage && !hasFiles) || APP.isGenerating) return;

      if (!APP.activeId) await newChat();

      // Clear hero if present
      const hero = UI.messagesContainer.querySelector('.llama-hero');
      if (hero) hero.remove();

      let text = rawText;
      let displayMsgText = rawText;
      if (hasFiles) {
        let fileBlock = 'Attached Context Files:\n';
        for (const f of APP.attachedFiles) {
          fileBlock += `--- BEGIN FILE: ${f.name} (${f.size} bytes) ---\n${f.content}\n--- END FILE: ${f.name} ---\n\n`;
        }
        text = fileBlock + (rawText || 'Please inspect the attached file(s).');
        const fileNamesList = APP.attachedFiles.map(f => `📄 ${escapeHtml(f.name)}`).join(', ');
        displayMsgText = (rawText ? `${rawText}\n\n` : '') + `*Attached files:* ${fileNamesList}`;
        APP.attachedFiles = [];
        renderAttachedFilesBar();
      }

      if (!APP.activeConv) APP.activeConv = { id: APP.activeId, messages: [] };
      if (!APP.activeConv.messages) APP.activeConv.messages = [];
      const userIdx = APP.activeConv.messages.length;
      APP.activeConv.messages.push({ role: 'user', content: text });

      appendMessage({ role: 'user', content: displayMsgText }, userIdx);
      isUserScrolledUp = false;
      if (UI.btnScrollBottom) UI.btnScrollBottom.classList.remove('visible');
      scrollDown(true);

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
      if (APP.activeConv && APP.activeConv.messages && APP.activeConv.messages.length <= 1) {
        const titleSource = rawText || (hasFiles ? 'Attached Files Analysis' : 'New Chat');
        renameChat(APP.activeId, titleSource.slice(0, 28) + (titleSource.length > 28 ? '...' : ''));
      }

      // Reset Input
      UI.chatInput.value = '';
      UI.chatInput.style.height = 'auto';
      const imageToSend = APP.attachedImage;
      UI.btnRemoveThumb.click();

      await triggerAssistantTurn(imageToSend);
    }

    async function resumeAssistantTurn(state) {
      if (APP.isGenerating) return;

      const { group, body, content, accordion, actions } = state;
      let assistantText = state.assistantText || '';
      let thoughtText = state.thoughtText || '';

      if (state.btnResume && state.btnResume.parentNode) {
        state.btnResume.remove();
        state.btnResume = null;
      }

      const notice = content.querySelector('.generation-stopped-notice');
      if (notice) notice.remove();

      APP.isGenerating = true;
      UI.btnSend.style.display = 'none';
      UI.btnStop.style.display = 'flex';
      UI.slotText.textContent = '1/1 slots (processing)';
      if (UI.footerSpeedMetrics) {
        UI.footerSpeedMetrics.textContent = 'generating...';
      }

      APP.abortController = new AbortController();
      let wasCancelled = false;
      let genTokens = 0;
      let startTime = performance.now();
      let accumulatedUsage = null;
      let accumulatedTimings = null;

      try {
        const reqModel = APP.modelId || (UI.modelPill.textContent !== 'loading model...' ? UI.modelPill.textContent : 'default');

        const msgs = [];
        if (APP.settings.system_prompt) {
          msgs.push({ role: 'system', content: APP.settings.system_prompt });
        }
        if (APP.activeConv && APP.activeConv.messages) {
          const allMsgs = APP.activeConv.messages;
          const lastIdx = allMsgs.length - 1;
          for (let i = 0; i < allMsgs.length; i++) {
            if (i === lastIdx && allMsgs[i].role === 'assistant') {
              continue;
            }
            msgs.push({ role: allMsgs[i].role, content: allMsgs[i].content });
          }
        }

        if (assistantText) {
          msgs.push({ role: 'assistant', content: assistantText });
        }

        msgs.push({
          role: 'user',
          content: 'Continue generating from exactly where you left off. Start immediately with the continuation, without repeating anything already written.'
        });

        const payload = {
          model: reqModel,
          messages: msgs,
          stream: true,
          stream_options: { include_usage: true },
          include_usage: true,
          temperature: Number(APP.settings.temperature) || 0.6,
          top_p: Number(APP.settings.top_p) || 0.95,
          max_tokens: parseInt(APP.settings.max_tokens, 10) || 4096
        };

        if (assistantText.length > 0) {
          payload.think = false;
          payload.reasoning_budget_tokens = 0;
          payload.chat_template_kwargs = { thinking: false };
        } else {
          let thinkBudget = null;
          if (APP.thinkingLevel === 'minimum') thinkBudget = 512;
          else if (APP.thinkingLevel === 'low') thinkBudget = 2048;
          else if (APP.thinkingLevel === 'medium') thinkBudget = 4096;
          else if (APP.thinkingLevel === 'high') thinkBudget = 8192;
          else if (APP.thinkingLevel === 'xhigh') {
            thinkBudget = 16384;
            payload.max_tokens = Math.max(payload.max_tokens, 16384);
          }
          if (thinkBudget) {
            payload.think = true;
            payload.reasoning_budget_tokens = thinkBudget;
            payload.thinking_budget_tokens = thinkBudget;
            payload.thinking = { type: 'enabled', budget_tokens: thinkBudget };
            payload.chat_template_kwargs = { thinking: true, thinking_budget: thinkBudget };
          }
        }

        payload.is_resume = true;
        const fetchUrl = APP.activeId ? `/api/conversations/${APP.activeId}/generate` : '/v1/chat/completions';
        const res = await fetch(fetchUrl, {
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
              if (chunk.usage) accumulatedUsage = chunk.usage;
              if (chunk.timings) accumulatedTimings = chunk.timings;

              const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
              if (delta) {
                if (delta.reasoning_content || delta.thought) {
                  const t = delta.reasoning_content || delta.thought;
                  thoughtText += t;
                  genTokens++;
                  updateLiveSpeed(genTokens, startTime);
                  if (accordion) {
                    accordion.style.display = 'block';
                    accordion.querySelector('.thought-text').textContent = thoughtText;
                    scrollDown();
                  }
                }

                if (delta.content) {
                  if (thoughtText && accordion && !accordion.querySelector('.thought-text').classList.contains('collapsed')) {
                    accordion.querySelector('.thought-text').classList.add('collapsed');
                    accordion.querySelector('span').innerHTML = '&#9654; Thought Process';
                    scrollDown(true);
                  }

                  assistantText += delta.content;
                  genTokens++;
                  updateLiveSpeed(genTokens, startTime);
                  content.innerHTML = parseMarkdown(assistantText);
                  scrollDown();
                }
              }
            } catch (e) {}
          }
        }
      } catch (err) {
        if (err.name === 'AbortError' || (APP.abortController && APP.abortController.signal && APP.abortController.signal.aborted)) {
          wasCancelled = true;
          const noticeEl = document.createElement('p');
          noticeEl.className = 'generation-stopped-notice';
          noticeEl.style.cssText = 'color:var(--text-dim); font-style:italic; margin-top:8px; font-size:0.85rem;';
          noticeEl.textContent = '[generation stopped]';
          content.appendChild(noticeEl);
        } else {
          content.innerHTML += `<p style="color:var(--danger); margin-top:8px;">[Error: ${escapeHtml(err.message)}]</p>`;
        }
      } finally {
        APP.isGenerating = false;
        APP.abortController = null;
        UI.btnSend.style.display = 'flex';
        UI.btnStop.style.display = 'none';
        UI.slotText.textContent = '1/1 slots (idle)';
        if (UI.footerSpeedMetrics && UI.footerSpeedMetrics.textContent === 'generating...') {
          UI.footerSpeedMetrics.textContent = 'eval: 0.0 t/s | prompt: 0.0 t/s';
        }

        // Update state references
        state.assistantText = assistantText;
        state.thoughtText = thoughtText;

        // Update copy button in actions
        const btnCopy = actions.querySelector('.btn-msg-action:not(.btn-resume)');
        if (btnCopy) {
          btnCopy.onclick = () => {
            navigator.clipboard.writeText(assistantText || '');
            btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
            setTimeout(() => {
              btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
            }, 1500);
          };
        }

        // Update activeConv & DB
        if (APP.activeConv && APP.activeConv.messages && APP.activeConv.messages.length > 0) {
          const lastIdx = APP.activeConv.messages.length - 1;
          if (APP.activeConv.messages[lastIdx].role === 'assistant') {
            APP.activeConv.messages[lastIdx].content = assistantText;
            APP.activeConv.messages[lastIdx].reasoning_content = thoughtText;
          } else {
            APP.activeConv.messages.push({
              role: 'assistant',
              content: assistantText,
              reasoning_content: thoughtText
            });
          }

          if (APP.activeId && !fetchUrl.includes('/api/conversations/')) {
            try {
              await apiFetch(`/api/conversations/${APP.activeId}/messages`, {
                method: 'PUT',
                body: { messages: APP.activeConv.messages }
              });
            } catch (_) {}
          }
        }

        // If it got cancelled AGAIN, show the resume button again!
        // If it completed normally, wasCancelled is false, so no resume button is shown!
        if (wasCancelled) {
          const newBtnResume = document.createElement('button');
          newBtnResume.className = 'btn-msg-action btn-resume';
          newBtnResume.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume';
          newBtnResume.title = 'Resume / Continue generation';
          newBtnResume.onclick = () => {
            state.btnResume = newBtnResume;
            resumeAssistantTurn(state);
          };
          actions.prepend(newBtnResume);
        }

        calculateContextSize();
        scrollDown();
      }
    }

    async function resumeLiveStream(convId, isResume = false) {
      if (APP.isGenerating) return;
      if (APP.activeId !== convId) return;

      APP.isGenerating = true;
      UI.btnSend.style.display = 'none';
      UI.btnStop.style.display = 'flex';
      UI.slotText.textContent = '1/1 slots (processing)';
      if (UI.footerSpeedMetrics) {
        UI.footerSpeedMetrics.textContent = 'generating...';
      }

      APP.abortController = new AbortController();

      let group = null;
      let body = null;
      let accordion = null;
      let content = null;
      let assistantText = '';
      let thoughtText = '';

      if (isResume) {
        const lastGroup = UI.messagesContainer.querySelector('.msg-group:last-child');
        if (lastGroup && lastGroup.classList.contains('assistant')) {
          group = lastGroup;
          body = group.querySelector('.msg-body');
          accordion = group.querySelector('.thought-accordion');
          content = group.querySelector('.md-content');
          if (APP.activeConv && APP.activeConv.messages && APP.activeConv.messages.length > 0) {
            const lastMsg = APP.activeConv.messages[APP.activeConv.messages.length - 1];
            if (lastMsg.role === 'assistant') {
              assistantText = lastMsg.content || '';
              thoughtText = lastMsg.reasoning_content || '';
            }
          }
        }
      }

      if (!group) {
        group = document.createElement('div');
        group.className = 'msg-group assistant';
        const author = document.createElement('div');
        author.className = 'msg-author assistant';
        author.textContent = 'Gufo';

        body = document.createElement('div');
        body.className = 'msg-body';

        accordion = document.createElement('div');
        accordion.className = 'thought-accordion';
        accordion.style.display = 'none';
        accordion.innerHTML = `
          <div class="thought-summary" onclick="toggleThoughtBlock(this)">
            <span>&#9660; Thinking...</span>
          </div>
          <div class="thought-text"></div>
        `;
        body.appendChild(accordion);

        content = document.createElement('div');
        content.className = 'md-content';
        body.appendChild(content);

        group.appendChild(author);
        group.appendChild(body);
        UI.messagesContainer.appendChild(group);
        scrollDown(true);
      }

      let startTime = performance.now();
      let genTokens = 0;
      let accumulatedUsage = null;
      let accumulatedTimings = null;
      let wasCancelled = false;

      try {
        const streamUrl = `/api/conversations/${convId}/stream`;
        const res = await fetch(streamUrl, {
          method: 'GET',
          headers: {
            ...(APP.token ? { 'Authorization': `Bearer ${APP.token}` } : {})
          },
          signal: APP.abortController.signal
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
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
              if (chunk.usage) accumulatedUsage = chunk.usage;
              if (chunk.timings) accumulatedTimings = chunk.timings;

              const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
              if (delta) {
                if (delta.reasoning_content || delta.thought) {
                  const t = delta.reasoning_content || delta.thought;
                  thoughtText += t;
                  genTokens++;
                  updateLiveSpeed(genTokens, startTime);
                  if (accordion) {
                    accordion.style.display = 'block';
                    accordion.querySelector('.thought-text').textContent = thoughtText;
                    scrollDown();
                  }
                }

                if (delta.content) {
                  if (thoughtText && accordion && !accordion.querySelector('.thought-text').classList.contains('collapsed')) {
                    accordion.querySelector('.thought-text').classList.add('collapsed');
                    accordion.querySelector('span').innerHTML = '&#9654; Thought Process';
                    scrollDown(true);
                  }

                  assistantText += delta.content;
                  genTokens++;
                  updateLiveSpeed(genTokens, startTime);
                  if (content) {
                    content.innerHTML = parseMarkdown(assistantText);
                    scrollDown();
                  }
                }
              }
            } catch (e) {}
          }
        }
      } catch (err) {
        if (err.name === 'AbortError' || (APP.abortController && APP.abortController.signal && APP.abortController.signal.aborted)) {
          wasCancelled = true;
          if (content) {
            content.innerHTML += '<p class="generation-stopped-notice" style="color:var(--text-dim); font-style:italic; margin-top:8px; font-size:0.85rem;">[generation stopped]</p>';
          }
        } else {
          if (content) {
            content.innerHTML += `<p style="color:var(--danger); margin-top:8px;">[Error: ${escapeHtml(err.message)}]</p>`;
          }
        }
      } finally {
        APP.isGenerating = false;
        APP.abortController = null;
        UI.btnSend.style.display = 'flex';
        UI.btnStop.style.display = 'none';
        UI.slotText.textContent = '1/1 slots (idle)';

        const oldActions = group.querySelector('.msg-actions');
        if (oldActions) oldActions.remove();

        const actions = document.createElement('div');
        actions.className = 'msg-actions';

        if (wasCancelled && (assistantText || thoughtText)) {
          const btnResume = document.createElement('button');
          btnResume.className = 'btn-msg-action btn-resume';
          btnResume.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume';
          btnResume.title = 'Resume / Continue generation';
          btnResume.onclick = () => {
            resumeAssistantTurn({
              group,
              body,
              content,
              accordion,
              actions,
              btnResume,
              assistantText,
              thoughtText
            });
          };
          actions.appendChild(btnResume);
        }

        const btnCopy = document.createElement('button');
        btnCopy.className = 'btn-msg-action';
        btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
        btnCopy.title = 'Copy message';
        btnCopy.onclick = () => {
          navigator.clipboard.writeText(assistantText || '');
          btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
          setTimeout(() => {
            btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
          }, 1500);
        };
        actions.appendChild(btnCopy);

        const finalCompTokens = accumulatedUsage ? (accumulatedUsage.completion_tokens ?? genTokens) : genTokens;
        const finalPromptTokens = accumulatedUsage ? (accumulatedUsage.prompt_tokens ?? 0) : 0;
        const finalCachedTokens = accumulatedUsage ? (accumulatedUsage.cached_tokens ?? (accumulatedUsage.prompt_tokens_details && accumulatedUsage.prompt_tokens_details.cached_tokens ? accumulatedUsage.prompt_tokens_details.cached_tokens : 0)) : 0;
        const finalTotalTokens = accumulatedUsage ? (accumulatedUsage.total_tokens ?? (finalPromptTokens + finalCompTokens)) : (finalPromptTokens + finalCompTokens);

        const evalTps = (accumulatedUsage && accumulatedUsage.completion_tokens_per_second)
          ? Number(accumulatedUsage.completion_tokens_per_second).toFixed(1)
          : (accumulatedTimings && accumulatedTimings.predicted_per_second)
            ? Number(accumulatedTimings.predicted_per_second).toFixed(1)
            : (finalCompTokens > 0 ? (finalCompTokens / Math.max(0.001, (performance.now() - startTime) / 1000)).toFixed(1) : null);

        const promptTps = (accumulatedUsage && accumulatedUsage.prompt_tokens_per_second)
          ? Number(accumulatedUsage.prompt_tokens_per_second).toFixed(1)
          : (accumulatedTimings && accumulatedTimings.prompt_per_second)
            ? Number(accumulatedTimings.prompt_per_second).toFixed(1)
            : null;

        if (finalCompTokens > 0 || finalPromptTokens > 0) {
          const usageBadge = document.createElement('div');
          usageBadge.className = 'msg-usage-badge';
          let badgeHtml = `<span class="usage-item" title="Completion tokens"><span class="usage-accent">${finalCompTokens}</span> tokens</span>`;
          if (finalPromptTokens > 0) {
            badgeHtml += ` &bull; <span class="usage-item" title="Prompt tokens"><span class="usage-val">${finalPromptTokens}</span> prompt</span>`;
            if (finalCachedTokens > 0) {
              badgeHtml += ` <span class="usage-item" style="color:var(--success);" title="Cached prompt tokens">(${finalCachedTokens} cached)</span>`;
            }
          }
          if (evalTps && Number(evalTps) > 0) {
            badgeHtml += ` &bull; <span class="usage-item" title="Generation speed"><span class="usage-val">${evalTps}</span> t/s</span>`;
          }
          if (promptTps && Number(promptTps) > 0) {
            badgeHtml += ` &bull; <span class="usage-item" title="Prefill speed"><span class="usage-val">${promptTps}</span> prompt t/s</span>`;
          }
          usageBadge.innerHTML = badgeHtml;
          actions.appendChild(usageBadge);
          if (convId) {
            const lastIdx = (APP.activeConv && APP.activeConv.messages) ? (APP.activeConv.messages.length - 1) : null;
            const curMsg = { id: null, content: assistantText };
            saveMessageBadge(convId, curMsg, lastIdx, badgeHtml);
            if (APP.activeConv && APP.activeConv.messages && lastIdx >= 0) {
              APP.activeConv.messages[lastIdx].usage_badge = badgeHtml;
            }
          }

          let footerText = `eval: ${evalTps || '0.0'} t/s`;
          if (promptTps && Number(promptTps) > 0) footerText += ` | prompt: ${promptTps} t/s`;
          if (finalCachedTokens > 0) footerText += ` | cached: ${finalCachedTokens}`;
          footerText += ` | total: ${finalTotalTokens} tokens`;
          UI.footerSpeedMetrics.textContent = footerText;
        }

        group.appendChild(actions);
        scrollDown();

        if (APP.activeConv && APP.activeConv.messages) {
          const lastIdx = APP.activeConv.messages.length - 1;
          if (lastIdx >= 0 && APP.activeConv.messages[lastIdx].role === 'assistant') {
            APP.activeConv.messages[lastIdx].content = assistantText;
            APP.activeConv.messages[lastIdx].reasoning_content = thoughtText;
          } else {
            APP.activeConv.messages.push({
              role: 'assistant',
              content: assistantText,
              reasoning_content: thoughtText
            });
          }
        }
        const item = APP.conversations.find(c => c.id === convId);
        if (item) item.is_generating = false;
        renderChatList();
      }
    }

    async function triggerAssistantTurn(imageToSend = null) {
      if (APP.isGenerating) return;

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

      if (imageToSend && msgs.length > 0) {
        const lastMsg = msgs[msgs.length - 1];
        if (lastMsg.role === 'user') {
          lastMsg.content = [
            { type: 'text', text: typeof lastMsg.content === 'string' ? lastMsg.content : '' },
            { type: 'image_url', image_url: { url: imageToSend } }
          ];
        }
      }

      const activeTools = getActiveToolsList();
      if (activeTools.some(t => t.function && t.function.name !== 'ask_user') && APP.settings.working_directory) {
        msgs.unshift({
          role: 'system',
          content: `Host server working directory: ${APP.settings.working_directory}`
        });
      }
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
          isUserScrolledUp = false;
          if (UI.btnScrollBottom) UI.btnScrollBottom.classList.remove('visible');
          scrollDown(true);

          let assistantText = '';
          let thoughtText = '';
          let startTime = performance.now();
          let genTokens = 0;
          let accumulatedToolCalls = [];
          let accumulatedUsage = null;
          let accumulatedTimings = null;
          let wasCancelled = false;

          if (UI.footerSpeedMetrics) {
            UI.footerSpeedMetrics.textContent = 'generating...';
          }

          APP.abortController = new AbortController();

          try {
            const reqModel = APP.modelId || (UI.modelPill.textContent !== 'loading model...' ? UI.modelPill.textContent : 'default');
            const payload = {
              model: reqModel,
              messages: msgs,
              stream: true,
              stream_options: { include_usage: true },
              include_usage: true,
              temperature: Number(APP.settings.temperature) || 0.6,
              top_p: Number(APP.settings.top_p) || 0.95,
              max_tokens: parseInt(APP.settings.max_tokens, 10) || 4096
            };
            let thinkBudget = null;
            if (APP.thinkingLevel === 'minimum') thinkBudget = 512;
            else if (APP.thinkingLevel === 'low') thinkBudget = 2048;
            else if (APP.thinkingLevel === 'medium') thinkBudget = 4096;
            else if (APP.thinkingLevel === 'high') thinkBudget = 8192;
            else if (APP.thinkingLevel === 'xhigh') {
              thinkBudget = 16384;
              payload.max_tokens = Math.max(payload.max_tokens, 16384);
            }

            if (thinkBudget) {
              payload.think = true;
              payload.reasoning_budget_tokens = thinkBudget;
              payload.thinking_budget_tokens = thinkBudget;
              payload.thinking = { type: 'enabled', budget_tokens: thinkBudget };
              payload.chat_template_kwargs = { thinking: true, thinking_budget: thinkBudget };
            } else {
              // auto
              payload.think = true;
              payload.reasoning_budget_tokens = -1;
              payload.chat_template_kwargs = { thinking: true };
            }
            if (activeTools.length > 0) {
              payload.tools = activeTools;
              payload.tool_choice = 'auto';
            }

            const fetchUrl = APP.activeId ? `/api/conversations/${APP.activeId}/generate` : '/v1/chat/completions';
            const res = await fetch(fetchUrl, {
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
                  if (chunk.usage) {
                    accumulatedUsage = chunk.usage;
                  }
                  if (chunk.timings) {
                    accumulatedTimings = chunk.timings;
                  }

                  const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
                  if (delta) {
                    if (delta.reasoning_content || delta.thought) {
                      const t = delta.reasoning_content || delta.thought;
                      thoughtText += t;
                      genTokens++;
                      updateLiveSpeed(genTokens, startTime);
                      accordion.style.display = 'block';
                      accordion.querySelector('.thought-text').textContent = thoughtText;
                      scrollDown();
                    }

                    if (delta.content) {
                      if (thoughtText && !accordion.querySelector('.thought-text').classList.contains('collapsed')) {
                        accordion.querySelector('.thought-text').classList.add('collapsed');
                        accordion.querySelector('span').innerHTML = '&#9654; Thought Process';
                        scrollDown(true);
                      }

                      assistantText += delta.content;
                      genTokens++;
                      updateLiveSpeed(genTokens, startTime);
                      content.innerHTML = parseMarkdown(assistantText);
                      scrollDown();
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
            if (err.name === 'AbortError' || (APP.abortController && APP.abortController.signal && APP.abortController.signal.aborted)) {
              wasCancelled = true;
              content.innerHTML += '<p class="generation-stopped-notice" style="color:var(--text-dim); font-style:italic; margin-top:8px; font-size:0.85rem;">[generation stopped]</p>';
            } else {
              content.innerHTML += `<p style="color:var(--danger); margin-top:8px;">[Error: ${escapeHtml(err.message)}]</p>`;
            }
          }

          // Add Message Actions & Token Usage Badge to the completed assistant turn
          const actions = document.createElement('div');
          actions.className = 'msg-actions';

          if (wasCancelled && (assistantText || thoughtText)) {
            const btnResume = document.createElement('button');
            btnResume.className = 'btn-msg-action btn-resume';
            btnResume.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume';
            btnResume.title = 'Resume / Continue generation';
            btnResume.onclick = () => {
              resumeAssistantTurn({
                group,
                body,
                content,
                accordion,
                actions,
                btnResume,
                assistantText,
                thoughtText
              });
            };
            actions.appendChild(btnResume);
          }

          const btnCopy = document.createElement('button');
          btnCopy.className = 'btn-msg-action';
          btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
          btnCopy.title = 'Copy message';
          btnCopy.onclick = () => {
            navigator.clipboard.writeText(assistantText || '');
            btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
            setTimeout(() => {
              btnCopy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
            }, 1500);
          };
          actions.appendChild(btnCopy);

          const finalCompTokens = accumulatedUsage ? (accumulatedUsage.completion_tokens ?? genTokens) : genTokens;
          const finalPromptTokens = accumulatedUsage ? (accumulatedUsage.prompt_tokens ?? 0) : 0;
          const finalCachedTokens = accumulatedUsage ? (accumulatedUsage.cached_tokens ?? (accumulatedUsage.prompt_tokens_details && accumulatedUsage.prompt_tokens_details.cached_tokens ? accumulatedUsage.prompt_tokens_details.cached_tokens : 0)) : 0;
          const finalTotalTokens = accumulatedUsage ? (accumulatedUsage.total_tokens ?? (finalPromptTokens + finalCompTokens)) : (finalPromptTokens + finalCompTokens);

          const evalTps = (accumulatedUsage && accumulatedUsage.completion_tokens_per_second)
            ? Number(accumulatedUsage.completion_tokens_per_second).toFixed(1)
            : (accumulatedTimings && accumulatedTimings.predicted_per_second)
              ? Number(accumulatedTimings.predicted_per_second).toFixed(1)
              : (finalCompTokens > 0 ? (finalCompTokens / Math.max(0.001, (performance.now() - startTime) / 1000)).toFixed(1) : null);

          const promptTps = (accumulatedUsage && accumulatedUsage.prompt_tokens_per_second)
            ? Number(accumulatedUsage.prompt_tokens_per_second).toFixed(1)
            : (accumulatedTimings && accumulatedTimings.prompt_per_second)
              ? Number(accumulatedTimings.prompt_per_second).toFixed(1)
              : null;

          APP.lastSeenGenTokens = finalCompTokens;
          APP.lastSeenPromptTokens = finalPromptTokens;
          if (evalTps) APP.lastSeenGenSpeed = Number(evalTps);
          if (promptTps) APP.lastSeenPromptSpeed = Number(promptTps);

          if (finalCompTokens > 0 || finalPromptTokens > 0) {
            const usageBadge = document.createElement('div');
            usageBadge.className = 'msg-usage-badge';
            let badgeHtml = `<span class="usage-item" title="Completion tokens"><span class="usage-accent">${finalCompTokens}</span> tokens</span>`;
            if (finalPromptTokens > 0) {
              badgeHtml += ` &bull; <span class="usage-item" title="Prompt tokens"><span class="usage-val">${finalPromptTokens}</span> prompt</span>`;
              if (finalCachedTokens > 0) {
                badgeHtml += ` <span class="usage-item" style="color:var(--success);" title="Cached prompt tokens">(${finalCachedTokens} cached)</span>`;
              }
            }
            if (evalTps && Number(evalTps) > 0) {
              badgeHtml += ` &bull; <span class="usage-item" title="Generation speed"><span class="usage-val">${evalTps}</span> t/s</span>`;
            }
            if (promptTps && Number(promptTps) > 0) {
              badgeHtml += ` &bull; <span class="usage-item" title="Prefill speed"><span class="usage-val">${promptTps}</span> prompt t/s</span>`;
            }
            if (accumulatedUsage && accumulatedUsage.draft_tokens > 0) {
              const accPct = ((accumulatedUsage.draft_tokens_accepted || 0) / accumulatedUsage.draft_tokens * 100).toFixed(0);
              badgeHtml += ` &bull; <span class="usage-item" title="Draft acceptance rate"><span class="usage-val">${accPct}%</span> draft acc</span>`;
            }
            usageBadge.innerHTML = badgeHtml;
            actions.appendChild(usageBadge);

            if (APP.activeId) {
              const lastIdx = (APP.activeConv && APP.activeConv.messages) ? (APP.activeConv.messages.length - 1) : null;
              const curMsg = { id: null, content: assistantText };
              saveMessageBadge(APP.activeId, curMsg, lastIdx, badgeHtml);
              if (APP.activeConv && APP.activeConv.messages && lastIdx >= 0) {
                APP.activeConv.messages[lastIdx].usage_badge = badgeHtml;
              }
            }

            // Update footer speed metrics
            let footerText = `eval: ${evalTps || '0.0'} t/s`;
            if (promptTps && Number(promptTps) > 0) footerText += ` | prompt: ${promptTps} t/s`;
            if (finalCachedTokens > 0) footerText += ` | cached: ${finalCachedTokens}`;
            footerText += ` | total: ${finalTotalTokens} tokens`;
            UI.footerSpeedMetrics.textContent = footerText;
          }

          group.appendChild(actions);
          scrollDown();

          // Check if model emitted text-based tool calls e.g. <tool_call>{"name":"...","arguments":{...}}</tool_call>
          if (activeTools.length > 0 && accumulatedToolCalls.length === 0 && assistantText.includes('<tool_call>')) {
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

          // Persist assistant message to memory & DB
          if (assistantText || thoughtText) {
            if (APP.activeConv && APP.activeConv.messages) {
              const lastIdx = APP.activeConv.messages.length - 1;
              if (lastIdx >= 0 && APP.activeConv.messages[lastIdx].role === 'assistant') {
                APP.activeConv.messages[lastIdx].content = assistantText;
                APP.activeConv.messages[lastIdx].reasoning_content = thoughtText;
              } else {
                APP.activeConv.messages.push({
                  role: 'assistant',
                  content: assistantText,
                  reasoning_content: thoughtText
                });
              }
            }
            if (APP.activeId && !fetchUrl.includes('/api/conversations/')) {
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

          // If generation was cancelled, or no tool calls were requested, we are done with the loop!
          if (wasCancelled || activeTools.length === 0 || accumulatedToolCalls.length === 0) {
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

            // 1. Interactive Model Inquiry (ask_user)
            if (tc.name === 'ask_user') {
              const answer = await askUserInquiry(tc, parsedArgs);
              const resultStr = JSON.stringify({ response: answer, answered: true });

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
              continue;
            }

            // If master tools are disabled and the model tried to run other tools, inform it
            if (!APP.toolsEnabled) {
              const resultStr = JSON.stringify({ error: "Tool execution is disabled by user setting." });
              msgs.push({
                role: 'tool',
                tool_call_id: tc.id,
                name: tc.name,
                content: resultStr
              });
              continue;
            }

            // 2. Permission check: Ask before running
            if (APP.toolPermission === 'ask') {
              const approved = await askToolApproval(tc, parsedArgs);
              if (!approved) {
                const resultStr = JSON.stringify({ error: "Tool execution denied by user." });
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
                continue;
              }
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
                <pre style="margin:0 0 6px 0; color:var(--text); white-space:pre-wrap; word-break:break-word; overflow-wrap:break-word;">${escapeHtml(JSON.stringify(parsedArgs, null, 2))}</pre>
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
                  working_directory: (APP.activeConv && APP.activeConv.working_directory) ? APP.activeConv.working_directory : '',
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
              resultDisplay.innerHTML = `<div style="color:var(--text-dim); margin-bottom:4px; font-size:0.75rem;">Result:</div><pre style="margin:0; color:${isSuccess ? 'var(--text)' : 'var(--danger)'}; white-space:pre-wrap; word-break:break-word; overflow-wrap:break-word;">${escapeHtml(resultStr)}</pre>`;
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
        if (UI.footerSpeedMetrics && UI.footerSpeedMetrics.textContent === 'generating...') {
          UI.footerSpeedMetrics.textContent = 'eval: 0.0 t/s | prompt: 0.0 t/s';
        }
        if (APP.activeId) {
          try {
            const res = await apiFetch(`/api/conversations/${APP.activeId}`);
            if (res.ok) APP.activeConv = await res.json();
          } catch (_) {}
        }
        calculateContextSize();
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
            include_usage: true,
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
          const u = data.usage || {};
          const t = data.timings || {};
          const genT = data.tokens_predicted ?? u.completion_tokens ?? 0;
          const promptT = data.tokens_evaluated ?? u.prompt_tokens ?? 0;
          const cachedT = data.tokens_cached ?? u.cached_tokens ?? 0;
          const evalTps = (t.predicted_per_second || u.completion_tokens_per_second) ? Number(t.predicted_per_second || u.completion_tokens_per_second).toFixed(1) : null;
          const promptTps = (t.prompt_per_second || u.prompt_tokens_per_second) ? Number(t.prompt_per_second || u.prompt_tokens_per_second).toFixed(1) : null;

          let statusText = `done: ${genT} tokens generated`;
          if (promptT > 0) statusText += ` | ${promptT} prompt`;
          if (cachedT > 0) statusText += ` (${cachedT} cached)`;
          if (evalTps) statusText += ` | eval: ${evalTps} t/s`;
          if (promptTps) statusText += ` | prompt: ${promptTps} t/s`;
          UI.compStatus.textContent = statusText;

          APP.lastSeenGenTokens = genT;
          APP.lastSeenPromptTokens = promptT;
          if (evalTps) APP.lastSeenGenSpeed = Number(evalTps);
          if (promptTps) APP.lastSeenPromptSpeed = Number(promptTps);
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
      try {
        const resSet = await apiFetch('/api/settings');
        if (resSet.ok) {
          const sData = await resSet.json();
          if (sData) APP.settings = { ...APP.settings, ...sData };
        }
      } catch (e) {}
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
      updateAttachPopupUI();
      await loadConversations();
      updateAttachPopupUI();
      updateHeroWorkdirDisplay();
      updateInputMetaBar();
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
