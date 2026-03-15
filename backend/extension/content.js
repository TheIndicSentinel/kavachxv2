/**
 * KavachX Browser Extension - Universal AI Platform Content Interceptor
 * INTERCEPT → EVALUATE → RELEASE
 *
 * This content script must:
 * - Reliably find prompt text across DOM + Shadow DOM.
 * - Block the original submission event until governance decision is returned.
 * - Only release the native action when Kavach returns PASS/ALERT/HUMAN_REVIEW.
 * - Hard-block on BLOCK decisions.
 */

(function () {
  console.log('🛡️ Kavach AI Shield: Intercept-Evaluate-Release active on', window.location.hostname);

  let lastInterceptedText = '';
  let lastInterceptTime = 0;

  let pendingSubmission = null; // { type: 'keyboard'|'click', originalEvent, triggerElement }

  /**
   * Robust node walker that crawls DOM + shadow roots to discover
   * visible, editable elements that are likely to contain prompts.
   */
  const findPromptNodeAndValue = () => {
    const selectors = [
      '#prompt-textarea',
      'textarea[placeholder*="Message"]',
      '.ProseMirror',
      'textarea[placeholder*="Talk to"]',
      '#editable-content-area',
      'textarea[placeholder*="Ask"]',
      'textarea[placeholder*="Type"]',
      'textarea[placeholder*="Send"]',
      '[role="textbox"]',
      '[contenteditable="true"]',
      'textarea',
      'input[type="text"]',
    ];

    const visitedRoots = new Set();

    const collectFromRoot = (root) => {
      if (!root || visitedRoots.has(root)) return [];
      visitedRoots.add(root);

      const hits = [];
      selectors.forEach((sel) => {
        try {
          root.querySelectorAll(sel).forEach((el) => {
            if (!el || el.offsetWidth === 0 || el.offsetHeight === 0) return;
            const val = (el.value || el.innerText || el.textContent || '').trim();
            if (val && val.length > 0) {
              hits.push({ el, text: val });
            }
          });
        } catch {
          // ignore selector errors
        }
      });

      // Recurse into any shadow roots
      root.querySelectorAll('*').forEach((n) => {
        if (n.shadowRoot) {
          hits.push(...collectFromRoot(n.shadowRoot));
        }
      });

      return hits;
    };

    let candidates = collectFromRoot(document);

    if (candidates.length === 0) {
      // Greedy fallback across visible editable containers
      const extra = [];
      const pushIfVisible = (node) => {
        if (!node || node.offsetWidth === 0 || node.offsetHeight === 0) return;
        const text = (node.innerText || node.value || '').trim();
        if (text && text.length > 0) {
          extra.push({ el: node, text });
        }
      };

      document.querySelectorAll('div[contenteditable], textarea').forEach(pushIfVisible);
      document.querySelectorAll('*').forEach((n) => {
        if (n.shadowRoot) {
          n.shadowRoot.querySelectorAll('div[contenteditable], textarea').forEach(pushIfVisible);
        }
      });

      candidates = extra;
    }

    if (!candidates.length) return { el: null, text: '' };

    // Prefer medium-length prompts; otherwise pick longest
    let best = candidates[0];
    for (const c of candidates) {
      if (c.text.length > 3 && c.text.length < 5000) {
        best = c;
        break;
      }
      if (c.text.length > best.text.length) best = c;
    }

    return { el: best.el, text: best.text };
  };

  const sendToKavach = (text, triggerMeta) => {
    if (!text) return;

    const now = Date.now();
    if (text === lastInterceptedText && now - lastInterceptTime < 1000) return;

    lastInterceptedText = text;
    lastInterceptTime = now;

    try {
      if (!chrome.runtime || !chrome.runtime.id) return;
      console.log('🛡️ KavachX: Intercepted prompt →', text.substring(0, 80) + '…');
      chrome.runtime.sendMessage({
        action: 'evaluate_prompt',
        prompt: text,
        domain: window.location.hostname,
        triggerMeta,
      });
    } catch {
      // ignore; if runtime is unavailable the original event will continue
    }
  };

  /**
   * Once a governance decision is received from background,
   * decide whether to release or permanently block the original event.
   */
  const handleGovernanceDecision = (decisionPayload) => {
    const { decision, explanation } = decisionPayload || {};
    if (!pendingSubmission) return;

    const original = pendingSubmission;
    pendingSubmission = null;

    if (decision === 'BLOCK') {
      console.warn('🛡️ KavachX: Prompt blocked by governance engine.', explanation?.reason);
      // Hard-block: do NOT re-dispatch the event. Optionally show inline banner.
      try {
        if (original.triggerElement) {
          original.triggerElement.dispatchEvent(
            new CustomEvent('kavachx:block', {
              bubbles: true,
              detail: { reason: explanation?.reason || 'Policy violation detected.' },
            }),
          );
        }
      } catch {
        // ignore
      }
      return;
    }

    // PASS / ALERT / HUMAN_REVIEW → allow original intent to proceed
    try {
      if (original.type === 'keyboard' && original.triggerElement) {
        // Synthesize a new Enter keydown that the page can handle normally
        const ev = new KeyboardEvent('keydown', {
          key: 'Enter',
          code: 'Enter',
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true,
        });
        original.triggerElement.dispatchEvent(ev);
      } else if (original.type === 'click' && original.triggerElement) {
        const ev = new MouseEvent('click', { bubbles: true, cancelable: true });
        original.triggerElement.dispatchEvent(ev);
      }
    } catch (e) {
      console.warn('KavachX: Failed to replay original event', e);
    }
  };

  // Listen for governance decisions coming back from background.js
  chrome.runtime.onMessage.addListener((msg, _sender, _sendResponse) => {
    if (msg && msg.action === 'governance_result') {
      handleGovernanceDecision(msg.payload);
    }
  });

  // 1. Universal Keyboard Listener (Enter keys) — PREVENT DEFAULT, then evaluate
  document.addEventListener(
    'keydown',
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        const active = document.activeElement;
        const { el, text } = findPromptNodeAndValue();
        const targetEl = active && (active.isContentEditable || active.tagName === 'TEXTAREA' || active.tagName === 'INPUT') ? active : el;

        if (text && targetEl) {
          // Intercept submission
          e.stopImmediatePropagation();
          e.preventDefault();
          pendingSubmission = { type: 'keyboard', originalEvent: e, triggerElement: targetEl };
          sendToKavach(text, { mode: 'keyboard', tag: targetEl.tagName });
        }
      }
    },
    true,
  );

  // 2. Universal Click Listener (Send buttons/icons) — PREVENT DEFAULT, then evaluate
  document.addEventListener(
    'click',
    (e) => {
      const target = e.target;
      if (!target) return;
      const clickable = target.closest('button, [role="button"], [data-testid*="send"], [aria-label*="Send"], [aria-label*="submit"]');
      if (!clickable) return;

      const { el, text } = findPromptNodeAndValue();
      if (text && el) {
        e.stopImmediatePropagation();
        e.preventDefault();
        pendingSubmission = { type: 'click', originalEvent: e, triggerElement: clickable };
        sendToKavach(text, { mode: 'click', tag: clickable.tagName });
      }
    },
    true,
  );

  // 3. MutationObserver placeholder to support future behavioral monitoring hooks
  const observer = new MutationObserver(() => {
    // Reserved: can emit additional telemetry if needed.
  });

  try {
    observer.observe(document.body, { childList: true, subtree: true });
  } catch {
    // ignore
  }
})();

