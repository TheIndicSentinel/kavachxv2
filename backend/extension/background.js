/**
 * KavachX Browser Extension — Background Service Worker (v4.0)
 *
 * Changes v3.9 → v4.0:
 *  - Fixed API port (8002 → 8001 to match the backend)
 *  - Dynamic injection into already-open tabs on install / Chrome startup
 *  - Engine health check every 30 s via chrome.alarms (persists across SW restarts)
 *  - Graceful offline mode: WARN instead of hard BLOCK when engine is unreachable
 *    (configurable — set kavach_fail_closed=true in storage to restore old behaviour)
 *  - Notification IDs added (required by some Chrome versions)
 *  - Network-intercept prompts from interceptor.js handled via evaluate_prompt
 *  - Engine-status query handler for popup
 */

// Default matches PORT in backend/.env — override via chrome.storage.local key "kavach_api_base"
const KAVACHX_DEFAULT_BASE = 'http://127.0.0.1:8002';

async function getApiBase() {
  try {
    const { kavach_api_base } = await chrome.storage.local.get(['kavach_api_base']);
    return (kavach_api_base || KAVACHX_DEFAULT_BASE).replace(/\/$/, '');
  } catch {
    return KAVACHX_DEFAULT_BASE;
  }
}

// Scripts to inject into already-open tabs.
// Order matters: MAIN world interceptor must run before ISOLATED world content.js.
const INJECT_SCRIPTS = [
  { files: ['interceptor.js'], world: 'MAIN'     },
  { files: ['content.js'],     world: 'ISOLATED'  },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function normalizePlatform(hostname) {
  const h = (hostname || '').toLowerCase();
  if (h.includes('chatgpt.com') || h.includes('chat.openai.com')) return 'chatgpt';
  if (h.includes('claude.ai'))                                     return 'claude';
  if (h.includes('gemini.google.com'))                             return 'gemini';
  if (h.includes('copilot.microsoft.com'))                         return 'copilot';
  if (h.includes('web.whatsapp.com'))                              return 'whatsapp';
  if (h.includes('mail.google.com'))                               return 'gmail';
  if (h.includes('notion.so'))                                     return 'notion';
  if (h.includes('github.com'))                                    return 'github-copilot';
  return 'universal-ai';
}

async function getSessionId() {
  const { kavach_session_id: id } = await chrome.storage.local.get(['kavach_session_id']);
  if (id) return id;
  const newId = self.crypto.randomUUID();
  await chrome.storage.local.set({ kavach_session_id: newId });
  return newId;
}

// ── Engine Health Check ───────────────────────────────────────────────────────

async function pingEngine() {
  try {
    const base = await getApiBase();
    const r = await fetch(`${base}/health`, { signal: AbortSignal.timeout(5000) });
    const online = r.ok;
    await chrome.storage.local.set({
      kavach_engine_online:     online,
      kavach_engine_last_ping:  Date.now(),
    });
    return online;
  } catch {
    await chrome.storage.local.set({ kavach_engine_online: false });
    return false;
  }
}

// Returns the offline decision based on the user's fail-mode preference.
// Default (fail-open): WARN — allows prompt through with a visible notice.
// Opt-in (fail-closed): BLOCK — same hard-block as before.
async function offlineDecision() {
  const { kavach_fail_closed } = await chrome.storage.local.get(['kavach_fail_closed']);
  if (kavach_fail_closed) {
    return {
      decision: 'BLOCK',
      reason:   'KavachX Governance Engine is offline. Access denied (fail-closed mode).',
    };
  }
  return {
    decision: 'WARN',
    reason:   'KavachX Engine unreachable — prompt allowed with caution. '
              + 'Enable fail-closed mode to block when offline.',
  };
}

// ── Dynamic Injection ─────────────────────────────────────────────────────────
// Injects content scripts into tabs that were already open when the extension
// was installed or Chrome was started. Without this, those tabs need a reload.

async function injectIntoExistingTabs() {
  let tabs = [];
  try {
    tabs = await chrome.tabs.query({});
  } catch (err) {
    console.warn('KavachX: tabs.query failed —', err.message);
    return;
  }
  for (const tab of tabs) {
    if (!tab.id || !tab.url) continue;
    // Skip chrome:// and extension pages — scripting API will reject these
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('about:')) continue;
    for (const script of INJECT_SCRIPTS) {
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files:  script.files,
          world:  script.world,
        });
      } catch {
        // Tab may not allow injection (PDF viewer, devtools, etc.) — skip silently
      }
    }
  }
}

// ── Evaluate Prompt ──────────────────────────────────────────────────────────

async function evaluatePrompt(prompt, platform, domain) {
  const sessionId = await getSessionId();
  const base      = await getApiBase();
  const response  = await fetch(`${base}/api/v1/governance/simulate`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id:   'kavach-sentinel-v1',
      session_id: sessionId,
      input_data: { prompt, source: 'extension', platform, domain },
      prediction: { text: 'Simulated Prediction' },
      confidence: 0.95,
      context: {
        domain:              'external_governance',
        platform,
        jurisdiction:        'IN',
        shadow_ai_detected:  true,
      },
    }),
    signal: AbortSignal.timeout(8000),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ── Lifecycle Listeners ───────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  injectIntoExistingTabs();
  // Schedule recurring health check (every 0.5 min = 30 s)
  chrome.alarms.create('kavach_health_check', { periodInMinutes: 0.5 });
  pingEngine();
});

chrome.runtime.onStartup.addListener(() => {
  injectIntoExistingTabs();
  pingEngine();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'kavach_health_check') pingEngine();
});

// ── Message Handler ───────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const { action } = request;

  // Engine status — queried by popup or content.js to show indicator
  if (action === 'get_engine_status') {
    chrome.storage.local
      .get(['kavach_engine_online', 'kavach_engine_last_ping'])
      .then(data => sendResponse({
        online:   data.kavach_engine_online  ?? null,
        lastPing: data.kavach_engine_last_ping ?? null,
      }));
    return true;
  }

  if (action !== 'evaluate_prompt') return false;

  let hostname = '';
  try {
    hostname = new URL(sender.tab?.url || `https://${request.domain || 'unknown'}/`).hostname;
  } catch {
    hostname = request.domain || 'unknown';
  }
  const platform = normalizePlatform(hostname);
  const prompt   = request.prompt || '';
  const source   = request.source || 'key-intercept';

  console.log(`🛡️ KavachX: Evaluating [${platform}] — source: ${source}`);

  (async () => {
    try {
      const result   = await evaluatePrompt(prompt, platform, hostname);
      const decision = result.enforcement_decision || 'PASS';
      const reason   = result.policy_violations?.[0]?.message || 'Policy check complete.';

      if (decision === 'BLOCK') {
        chrome.notifications.create(`kavach-${Date.now()}`, {
          type:     'basic',
          iconUrl:  'icons/icon128.png',
          title:    '🚨 Kavach — BLOCKED',
          message:  `${platform.toUpperCase()}: ${reason.substring(0, 120)}`,
          priority: 2,
        });
      }

      // Update engine-online status on successful call
      chrome.storage.local.set({ kavach_engine_online: true, kavach_engine_last_ping: Date.now() });
      sendResponse({ decision, reason });

    } catch (err) {
      console.warn('🛡️ KavachX: Engine unreachable —', err.message);
      chrome.storage.local.set({ kavach_engine_online: false });
      const fallback = await offlineDecision();
      sendResponse(fallback);
    }
  })();

  return true; // Keep the async message channel open
});
