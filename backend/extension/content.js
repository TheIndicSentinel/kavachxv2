/**
 * KavachX Browser Extension — Precision Interceptor (v8)
 *
 * v7 → v8:
 * - postMessage bridge for interceptor.js (MAIN world) — routes LLM network
 *   requests through the background service worker and returns the decision.
 * - WebSocket AI-stream notification (passive — shows a subtle indicator).
 * - Soft WARN banner for offline / network-intercept non-blocking alerts.
 * - show_block handler wired to showBanner() for network-intercepted BLOCKs.
 */

(function () {
  'use strict';

  if (window.__kavachxInjectedV8) return;
  window.__kavachxInjectedV8 = true;

  console.log('🛡️ Kavach AI Shield v8: Precision Interceptor Active');

  let evaluating = false; // True while waiting for backend response
  let bypassing  = false; // True during the release window (prevents re-interception)
  let contextDead = false; // True after the extension context is invalidated

  // ── Context validity guard ─────────────────────────────────────────────────
  // After an extension reload/update, chrome.runtime.id becomes undefined and
  // any call to chrome.runtime.sendMessage throws "Extension context invalidated".
  // We detect this and tear down all listeners so the dead script stops interfering.

  function isContextValid() {
    try {
      return !!(chrome?.runtime?.id);
    } catch {
      return false;
    }
  }

  // Refs held for teardown
  const _domListeners = [];
  const _winListeners = [];

  function teardown() {
    if (contextDead) return;
    contextDead = true;
    console.log('🛡️ [Kavach] Extension context invalidated — removing listeners.');
    _domListeners.forEach(([type, fn]) => document.removeEventListener(type, fn, true));
    _winListeners.forEach(fn => window.removeEventListener('message', fn));
    // Reset the guard so a freshly injected v8 can re-attach
    window.__kavachxInjectedV8 = false;
  }

  // Safe wrapper: calls fn() only if context is still valid; tears down if not
  function safeRuntime(fn) {
    if (!isContextValid()) { teardown(); return; }
    try {
      fn();
    } catch (err) {
      if (err?.message?.includes('Extension context invalidated') ||
          err?.message?.includes('Cannot read properties of undefined')) {
        teardown();
      } else {
        console.warn('🛡️ [Kavach] Runtime error:', err.message);
      }
    }
  }

  // ─── Shadow-DOM Aware Prompt Scraper ──────────────────────────────────────
  function findPrompt() {
    const SELECTORS = [
      '#prompt-textarea',
      'div[contenteditable="true"]',
      '.ProseMirror',
      '[role="textbox"]',
      'textarea'
    ];
    const findInRoot = (root) => {
      for (const s of SELECTORS) {
        const el = root.querySelector(s);
        if (el && el.offsetParent !== null) {
          const val = (el.value || el.innerText || el.textContent || '').trim();
          if (val.length > 2) return { el, text: val };
        }
      }
      for (const child of root.querySelectorAll('*')) {
        if (child.shadowRoot) {
          const found = findInRoot(child.shadowRoot);
          if (found) return found;
        }
      }
      return null;
    };
    return findInRoot(document);
  }

  // ─── Find the real Send button ────────────────────────────────────────────
  function findSendButton() {
    // ChatGPT, Claude, Gemini, Copilot — ordered by specificity
    const SEND_SELECTORS = [
      '[data-testid="send-button"]',
      'button[aria-label="Send prompt"]',
      'button[aria-label="Send message"]',
      'button[aria-label="Send"]',
      'button[type="submit"]',
      'button.send-button',
      'form button:last-of-type',
    ];
    for (const s of SEND_SELECTORS) {
      const btn = document.querySelector(s);
      if (btn && !btn.disabled) return btn;
    }
    return null;
  }

  // ─── Release (PASS / ALERT) ───────────────────────────────────────────────
  function releasePrompt() {
    bypassing = true;

    // Strategy 1: click the real send button
    const sendBtn = findSendButton();
    if (sendBtn) {
      console.log('🛡️ [Kavach] Releasing via Send button click');
      sendBtn.click();
      setTimeout(() => { bypassing = false; }, 800);
      return;
    }

    // Strategy 2: requestSubmit() on the form
    const form = document.querySelector('form');
    if (form) {
      console.log('🛡️ [Kavach] Releasing via form.requestSubmit()');
      try { form.requestSubmit(); } catch (_) { form.submit(); }
      setTimeout(() => { bypassing = false; }, 800);
      return;
    }

    // Strategy 3: dispatch Enter on document.activeElement
    console.log('🛡️ [Kavach] Releasing via Enter on active element');
    const target = document.activeElement || document.body;
    target.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Enter', code: 'Enter', keyCode: 13,
      which: 13, bubbles: true, cancelable: false
    }));
    setTimeout(() => { bypassing = false; }, 800);
  }

  // ─── Security Banner ──────────────────────────────────────────────────────
  function showBanner(msg) {
    document.getElementById('kavachx-block-alert')?.remove();
    const b = document.createElement('div');
    b.id = 'kavachx-block-alert';
    b.style.cssText = [
      'position:fixed', 'top:30px', 'left:50%', 'transform:translateX(-50%)',
      'background:#0f0f0f', 'color:#f87171', 'padding:18px 28px',
      'border-radius:12px', 'z-index:2147483647', 'font-family:system-ui,sans-serif',
      'box-shadow:0 12px 40px rgba(0,0,0,0.8)', 'border:2px solid #ef4444',
      'text-align:center', 'max-width:440px', 'line-height:1.5'
    ].join(';');
    b.innerHTML = `
      <strong>🚨 KAVACH SECURITY BLOCK</strong>
      <div style="margin-top:8px;font-size:14px;opacity:0.92;">${msg}</div>
    `;
    document.body.appendChild(b);
    setTimeout(() => {
      b.style.transition = 'opacity 0.5s';
      b.style.opacity = '0';
      setTimeout(() => b.remove(), 500);
    }, 7000);
  }

  // ─── Main Interceptor ─────────────────────────────────────────────────────
  function handler(e) {
    // GUARD 1: We are in the release window — let the event through untouched
    if (bypassing) return;

    // GUARD 2: Debounce — we are already waiting for a backend response
    if (evaluating) {
      e.preventDefault();
      e.stopImmediatePropagation();
      return;
    }

    // Key filter: only Enter without Shift
    if (e.type === 'keydown' && (e.key !== 'Enter' || e.shiftKey)) return;

    // Click filter: only the Send button area
    if (['click', 'mousedown', 'pointerdown'].includes(e.type)) {
      const btn = e.target.closest('button, [role="button"], [data-testid*="send"], svg');
      if (!btn) return;
    }

    const result = findPrompt();
    if (!result || result.text.length < 3) return;

    // INTERCEPT
    e.preventDefault();
    e.stopImmediatePropagation();
    evaluating = true;

    console.log(`🛡️ [Kavach] Intercepted: "${result.text.substring(0, 45)}..."`);

    safeRuntime(() => {
      chrome.runtime.sendMessage(
        { action: 'evaluate_prompt', prompt: result.text, domain: window.location.hostname },
        (response) => {
          evaluating = false;

          // Suppress the "message channel closed" benign error Chrome emits on SW restart
          if (chrome.runtime.lastError) {
            const msg = chrome.runtime.lastError.message || '';
            if (msg.includes('context invalidated') || msg.includes('Extension context')) {
              teardown(); return;
            }
            // Engine offline — apply graceful fallback
            releasePrompt();
            return;
          }

          if (!response) { releasePrompt(); return; }

          const decision = (response.decision || 'PASS').toUpperCase();
          console.log(`🛡️ [Kavach] Decision: ${decision}`);

          if (decision === 'BLOCK') {
            showBanner(response.reason || 'Blocked by KavachX Governance Policy');
          } else if (decision === 'WARN') {
            showWarn(response.reason || 'AI request allowed with a governance warning.');
            releasePrompt();
          } else {
            // PASS / ALERT → release silently; audit log written to dashboard
            releasePrompt();
          }
        }
      );
    });
    if (contextDead) { evaluating = false; releasePrompt(); }
  }

  // Register capture-phase listeners and keep refs for teardown
  ['keydown', 'click', 'mousedown'].forEach(type => {
    _domListeners.push([type, handler]);
    document.addEventListener(type, handler, true);
  });

  // ─── Soft Warning Banner (WARN decision / offline) ────────────────────────
  function showWarn(msg) {
    document.getElementById('kavachx-warn-alert')?.remove();
    const b = document.createElement('div');
    b.id = 'kavachx-warn-alert';
    b.style.cssText = [
      'position:fixed', 'top:30px', 'left:50%', 'transform:translateX(-50%)',
      'background:#1c1a00', 'color:#fbbf24', 'padding:14px 22px',
      'border-radius:10px', 'z-index:2147483647', 'font-family:system-ui,sans-serif',
      'box-shadow:0 8px 32px rgba(0,0,0,0.7)', 'border:1.5px solid #f59e0b',
      'text-align:center', 'max-width:420px', 'line-height:1.5', 'font-size:13px',
    ].join(';');
    b.innerHTML = `<strong>⚠️ KAVACH WARNING</strong>
      <div style="margin-top:6px;opacity:0.9;">${msg}</div>`;
    document.body.appendChild(b);
    setTimeout(() => {
      b.style.transition = 'opacity 0.4s';
      b.style.opacity = '0';
      setTimeout(() => b.remove(), 400);
    }, 5000);
  }

  // ─── postMessage Bridge (MAIN world ↔ ISOLATED world ↔ background) ───────
  // interceptor.js (MAIN world) cannot call chrome.runtime.sendMessage directly.
  // It posts to window; we receive here and proxy to the background service worker,
  // then post the response back so interceptor.js can resolve its Promise.

  const _bridgeListener = (evt) => {
    if (evt.source !== window) return;
    if (contextDead) return;
    const msg = evt.data;
    if (!msg || typeof msg !== 'object') return;

    // ── Route LLM network-intercept evaluation ──────────────────────────────
    if (msg.kavachx_type === 'evaluate_request' && msg.prompt) {
      const msgId = msg.kavachx_msg_id;
      safeRuntime(() => {
      chrome.runtime.sendMessage(
        {
          action: 'evaluate_prompt',
          prompt: msg.prompt,
          domain: msg.domain || window.location.hostname,
          source: msg.source || 'network_intercept',
        },
        (response) => {
          if (chrome.runtime.lastError) {
            const errMsg = chrome.runtime.lastError.message || '';
            if (errMsg.includes('context invalidated') || errMsg.includes('Extension context')) {
              teardown(); return;
            }
            // Engine unreachable — fail open for network intercepts
            window.postMessage({
              kavachx_type:   'evaluate_response',
              kavachx_msg_id: msgId,
              decision:       'WARN',
              reason:         'KavachX Engine unreachable — request allowed with caution.',
            }, '*');
            showWarn('KavachX Engine unreachable — AI request allowed with caution.');
            return;
          }
          if (!response) {
            window.postMessage({ kavachx_type: 'evaluate_response', kavachx_msg_id: msgId, decision: 'PASS' }, '*');
            return;
          }
          window.postMessage({
            kavachx_type:   'evaluate_response',
            kavachx_msg_id: msgId,
            decision:       response.decision,
            reason:         response.reason,
          }, '*');
          if ((response.decision || '').toUpperCase() === 'WARN') {
            showWarn(response.reason || 'AI request allowed with a governance warning.');
          }
        }
      );
      }); // end safeRuntime
    }

    // ── Display block banner triggered by network-intercepted BLOCK ──────────
    if (msg.kavachx_type === 'show_block') {
      showBanner(msg.reason || 'Blocked by KavachX Governance Policy');
    }

    // ── Log passive WebSocket AI-stream detection ────────────────────────────
    if (msg.kavachx_type === 'ws_ai_stream') {
      console.log(`🛡️ [Kavach] AI streaming response detected on ${msg.domain}`);
    }
  };

  // Register and track for teardown
  _winListeners.push(_bridgeListener);
  window.addEventListener('message', _bridgeListener);

})();
