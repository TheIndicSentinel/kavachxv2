/**
 * KavachX — MAIN World Network Interceptor (v1.0)
 *
 * Runs in the PAGE's execution context (MAIN world) to intercept:
 *   1. window.fetch() — detects LLM API payloads by "LLM-DNA" JSON key
 *      fingerprinting, regardless of which domain the call targets.
 *   2. window.WebSocket — passively detects AI streaming responses for logging.
 *      (Blocking WebSocket mid-stream is intentionally NOT implemented — it
 *       causes unrecoverable page state in many SPAs.)
 *
 * Communication with content.js (ISOLATED world) uses window.postMessage.
 * This file does NOT have access to chrome.* APIs.
 *
 * LLM-DNA fingerprinting:
 *   Any POST body containing ≥ 2 of these keys is treated as an LLM API call:
 *   [prompt, messages, temperature, max_tokens, model, stream, top_p, n,
 *    frequency_penalty, presence_penalty]
 *
 *   This catches calls to OpenAI, Anthropic, Google Gemini, Mistral, Azure
 *   OpenAI, and any custom LLM proxy — regardless of the hostname.
 */

(function () {
  'use strict';

  if (window.__kavachxNetworkInterceptor) return;
  window.__kavachxNetworkInterceptor = true;

  // ── LLM fingerprint keys ────────────────────────────────────────────────────
  const LLM_KEYS = new Set([
    'prompt', 'messages', 'temperature', 'max_tokens', 'model',
    'stream', 'n', 'top_p', 'frequency_penalty', 'presence_penalty',
  ]);
  const LLM_MIN_SCORE = 2; // minimum matching keys to classify as LLM call

  // ── Helpers ─────────────────────────────────────────────────────────────────

  function scoreLlmBody(bodyStr) {
    try {
      const parsed = JSON.parse(bodyStr);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return { score: 0, parsed: null };
      }
      const score = [...LLM_KEYS].filter(k => k in parsed).length;
      return { score, parsed };
    } catch {
      return { score: 0, parsed: null };
    }
  }

  function extractPromptText(parsed) {
    // Direct "prompt" string
    if (typeof parsed.prompt === 'string' && parsed.prompt.length > 3) {
      return parsed.prompt;
    }
    // OpenAI / Anthropic messages array — extract the last user message
    if (Array.isArray(parsed.messages)) {
      const userMsgs = parsed.messages.filter(
        m => m?.role === 'user' || m?.role === 'human'
      );
      const last = userMsgs[userMsgs.length - 1];
      if (last?.content) {
        if (typeof last.content === 'string') return last.content;
        if (Array.isArray(last.content)) {
          return last.content.map(c => c?.text || '').join(' ').trim();
        }
      }
    }
    return '';
  }

  // Ask content.js (ISOLATED world) to route to the background service worker.
  // Returns a Promise<{decision, reason}> that resolves within 5 s.
  function askContentScript(prompt) {
    return new Promise((resolve) => {
      const msgId = `kavachx_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

      const cleanup = () => window.removeEventListener('message', onReply);

      const onReply = (evt) => {
        if (evt.source !== window) return;
        if (evt.data?.kavachx_type === 'evaluate_response' && evt.data?.kavachx_msg_id === msgId) {
          cleanup();
          resolve(evt.data);
        }
      };

      window.addEventListener('message', onReply);

      // Safety timeout: if content.js doesn't respond in 5 s, allow through
      setTimeout(() => {
        cleanup();
        resolve({ decision: 'PASS', reason: 'timeout' });
      }, 5000);

      window.postMessage({
        kavachx_type:   'evaluate_request',
        kavachx_msg_id: msgId,
        prompt,
        domain:         window.location.hostname,
        source:         'network_intercept',
      }, '*');
    });
  }

  // ── fetch() Interceptor ──────────────────────────────────────────────────────
  // Wraps window.fetch to inspect POST bodies for LLM API payloads.
  // On BLOCK decision: throws a named error so the page's .catch() surfaces it.
  // On PASS / WARN / timeout: proceeds with the original fetch unchanged.

  const _origFetch = window.fetch.bind(window);

  window.fetch = async function kavachxFetch(resource, init) {
    if (init?.method?.toUpperCase() === 'POST' && typeof init?.body === 'string') {
      const { score, parsed } = scoreLlmBody(init.body);
      if (score >= LLM_MIN_SCORE && parsed) {
        const text = extractPromptText(parsed);
        if (text.length > 3) {
          let result;
          try {
            result = await askContentScript(text);
          } catch {
            // postMessage bridge failure — fail open, don't block the request
            result = { decision: 'PASS' };
          }

          if (result?.decision === 'BLOCK') {
            // Notify content.js to show the block banner in the page
            window.postMessage({ kavachx_type: 'show_block', reason: result.reason }, '*');
            const err = new Error(
              `KavachX Governance: Request blocked by policy. ${result.reason || ''}`
            );
            err.name = 'KavachXBlockedError';
            throw err;
          }
          // WARN: let the request proceed — content.js shows a soft warning banner
        }
      }
    }
    return _origFetch(resource, init);
  };

  // ── XMLHttpRequest Interceptor ───────────────────────────────────────────────
  // Wraps XHR .open() and .send() to inspect POST bodies for LLM API payloads.
  // XHR is synchronous at the call site, so we capture the body in .send(),
  // run an async evaluation, then either call the original send (PASS) or
  // dispatch a ProgressEvent('error') to abort the request (BLOCK).
  //
  // State tracking uses a WeakMap — no property is added to the XHR instance.

  const _xhrState  = new WeakMap(); // xhr → { method, url, body }
  const _origOpen  = XMLHttpRequest.prototype.open;
  const _origSend  = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function kavachxOpen(method, url, ...rest) {
    _xhrState.set(this, { method: (method || '').toUpperCase(), url: url || '' });
    return _origOpen.call(this, method, url, ...rest);
  };

  XMLHttpRequest.prototype.send = function kavachxSend(body) {
    const state = _xhrState.get(this);

    // Only inspect POST requests with a string body
    if (!state || state.method !== 'POST' || typeof body !== 'string') {
      return _origSend.call(this, body);
    }

    const { score, parsed } = scoreLlmBody(body);
    if (score < LLM_MIN_SCORE || !parsed) {
      return _origSend.call(this, body);
    }

    const text = extractPromptText(parsed);
    if (text.length <= 3) {
      return _origSend.call(this, body);
    }

    // Intercept: run async evaluation, then release or abort
    const xhr = this;
    (async () => {
      let result;
      try {
        result = await askContentScript(text);
      } catch {
        result = { decision: 'PASS' };
      }

      if (result?.decision === 'BLOCK') {
        window.postMessage({ kavachx_type: 'show_block', reason: result.reason }, '*');
        // Dispatch a network error so the caller's onerror/onloadend fires
        xhr.dispatchEvent(new ProgressEvent('error', { bubbles: false, cancelable: false }));
      } else {
        // PASS or WARN — proceed with the original request
        _origSend.call(xhr, body);
      }
    })();

    // Return undefined synchronously (the async IIFE will handle the request)
  };

  // ── WebSocket Interceptor (passive — detection only) ────────────────────────
  // Observes WebSocket messages for AI streaming patterns (OpenAI, Anthropic).
  // Does NOT close or pause the socket — that would corrupt page state.

  const _OrigWebSocket = window.WebSocket;

  window.WebSocket = new Proxy(_OrigWebSocket, {
    construct(Target, args) {
      const ws = new Target(...args);
      ws.addEventListener('message', (evt) => {
        if (typeof evt.data !== 'string') return;
        try {
          const data = JSON.parse(evt.data);
          // OpenAI streaming: { choices: [{ delta: { content: "..." } }] }
          // Anthropic streaming: { type: "content_block_delta", delta: { text: "..." } }
          const isAiStream =
            data?.choices ||
            data?.delta    ||
            data?.type === 'content_block_delta' ||
            data?.type === 'message_delta';
          if (isAiStream) {
            window.postMessage({
              kavachx_type: 'ws_ai_stream',
              domain: window.location.hostname,
            }, '*');
          }
        } catch { /* Non-JSON WebSocket frame — ignore */ }
      });
      return ws;
    },
  });

})();
