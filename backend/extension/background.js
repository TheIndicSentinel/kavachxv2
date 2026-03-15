/**
 * KavachX Browser Extension - Background Service Worker
 * Communicates with the hosted KavachX Governance Engine.
 * Supports: ChatGPT, Claude, Gemini, Copilot, and other web AI platforms.
 *
 * This worker enforces the INTERCEPT → EVALUATE → RELEASE workflow by:
 * - Receiving intercepted prompts from content scripts.
 * - Calling the KavachX governance simulate endpoint.
 * - Sending a governance_result back to the originating tab/frame.
 */

// Replace this with your actual Render URL after deployment.
// No API key is required — the governance engine is designed to accept
// trusted traffic from the browser extension directly.
const KAVACH_SERVER_URL = 'http://localhost:8005';

// Map browser hostnames to human-readable platform names
const PLATFORM_MAP = {
  'chatgpt.com': 'ChatGPT',
  'chat.openai.com': 'ChatGPT',
  'claude.ai': 'Claude',
  'gemini.google.com': 'Gemini',
  'copilot.microsoft.com': 'Microsoft Copilot',
  'github.com': 'GitHub Copilot',
  'perplexity.ai': 'Perplexity',
  'deepseek.com': 'DeepSeek',
  'chat.mistral.ai': 'Mistral',
  'huggingface.co': 'HuggingFace',
  'poe.com': 'Poe',
};

function getPlatformName(hostname) {
  if (!hostname) return 'Universal AI';
  for (const [key, name] of Object.entries(PLATFORM_MAP)) {
    if (hostname.includes(key)) return name;
  }
  const parts = hostname.split('.');
  if (parts.length >= 2) return parts.slice(-2).join('.');
  return hostname || 'Universal AI';
}

let sessionId = null;

// Initialize session ID on startup
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(['sessionId'], (result) => {
    if (!result.sessionId) {
      sessionId = self.crypto.randomUUID();
      chrome.storage.local.set({ sessionId });
    } else {
      sessionId = result.sessionId;
    }
  });
});

// Helper to get session ID
const getSessionId = async () => {
  if (sessionId) return sessionId;
  const res = await chrome.storage.local.get(['sessionId']);
  return res.sessionId;
};

chrome.runtime.onMessage.addListener((request, sender, _sendResponse) => {
  if (request.action === 'evaluate_prompt') {
    const tabId = sender.tab && sender.tab.id;
    const frameId = sender.frameId;
    processGovernance(request.prompt, request.domain, { tabId, frameId });
  }
});

async function processGovernance(prompt, domain, origin) {
  const platform = getPlatformName(domain);
  let result = null;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);

    const response = await fetch(`${KAVACH_SERVER_URL}/api/v1/governance/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_id: 'kavach-sentinel-v1',
        session_id: await getSessionId(),
        input_data: {
          prompt: prompt,
          source: 'browser_extension',
          platform: platform,
        },
        prediction: { text: 'Pending Kavach Review' },
        confidence: 0.95,
        context: {
          domain: 'external_governance',
          browser_source: domain,
          platform: platform,
        },
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);
    result = await response.json();
  } catch (error) {
    console.error('❌ Kavach Governance Connection Error:', error);
    // On failure, default to safest reasonable behavior: REQUIRE HUMAN REVIEW.
    result = {
      enforcement_decision: 'HUMAN_REVIEW',
      explanation: {
        reason: 'Kavach governance service unreachable — defaulting to human review.',
        policy_triggered: 'KavachX Connectivity Guard',
      },
    };
  }

  await notifyTabWithDecision(result, prompt, platform, origin);
}

async function notifyTabWithDecision(result, prompt, platform, origin) {
  const decision = result.enforcement_decision || 'PASS';
  const explanation = result.explanation || {};

  if (decision === 'BLOCK') {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: '🚨 Kavach Security Alert',
      message: `Prompt BLOCKED on ${platform}: Policy violation detected.`,
      priority: 2,
    });
  } else if (decision === 'HUMAN_REVIEW') {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: '⚠️ Kavach Review Required',
      message: `Prompt on ${platform} flagged for manual governance review.`,
      priority: 1,
    });
  } else if (decision === 'ALERT') {
    console.warn(`⚠️ Kavach Alert on ${platform}:`, (prompt || '').substring(0, 80));
  } else {
    console.log(`✅ Kavach PASS on ${platform}: No policy violation.`);
  }

  if (origin && typeof origin.tabId === 'number') {
    try {
      chrome.tabs.sendMessage(origin.tabId, {
        action: 'governance_result',
        payload: { decision, explanation },
      });
    } catch (e) {
      console.warn('KavachX: Failed to send governance_result to tab', e);
    }
  }
}

