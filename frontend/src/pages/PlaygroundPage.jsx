import { useState } from 'react'
import Card from '../components/shared/Card'
import { Send, Shield, Zap, AlertTriangle, CheckCircle, Terminal, Search, Settings } from 'lucide-react'
import { governanceAPI } from '../utils/api'

export default function PlaygroundPage() {
  const [prompt, setPrompt] = useState('')
  const [selectedContext, setSelectedContext] = useState('general')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const CONTEXTS = [
    { id: 'general', label: 'General / Safery', icon: Shield, color: 'var(--accent)', desc: 'Standard safety & toxicity checks.' },
    { id: 'finance', label: 'Finance (RBI)', icon: Zap, color: 'var(--cyan)', desc: 'RBI Fair Lending & DTI checks.' },
    { id: 'healthcare', label: 'Healthcare (NHA)', icon: Activity, color: 'var(--red)', desc: 'NHA & ABDM Privacy checks.' },
    { id: 'education', label: 'Edu (NEP 2020)', icon: Info, color: 'var(--purple)', desc: 'EdTech Surveillance checks.' },
  ]

  // Intelligent detection for manual prompts
  const getContextPayload = (ctx) => {
    switch(ctx) {
      case 'finance': return { input_data: { caste_proxy_score: 0.15, debt_ratio: 0.45 }, context: { domain: 'finance', regulation: 'RBI' } }
      case 'healthcare': return { input_data: { personal_data_used: true, consent_verified: false }, context: { domain: 'healthcare', abdm_linked: true } }
      case 'education': return { input_data: { continuous_monitoring: true, parental_consent: false }, context: { domain: 'education' } }
      default: return { input_data: { toxicity_score: 0.85, prompt_injection_score: 0.92 }, context: { domain: 'general' } }
    }
  }

  const handleTestManual = async (e) => {
    e?.preventDefault()
    if (!prompt.trim()) return
    
    setLoading(true)
    setResult(null)
    const payload = getContextPayload(selectedContext)
    
    try {
      const response = await governanceAPI.simulate({
        input_data: { ...payload.input_data, prompt: prompt },
        prediction: { content: "Evaluating your prompt against GaaS policy..." },
        context: payload.context,
        confidence: 0.9
      })
      setResult({ input: prompt, response: response.data })
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (d) => {
    if (d === 'BLOCK') return 'var(--red)'
    if (d === 'ALERT') return 'var(--amber)'
    if (d === 'HUMAN_REVIEW') return 'var(--purple)'
    return 'var(--green)'
  }

  const getStatusIcon = (d) => {
    if (d === 'BLOCK') return <Shield size={18} color="var(--red)"/>
    if (d === 'ALERT') return <AlertTriangle size={18} color="var(--amber)"/>
    if (d === 'HUMAN_REVIEW') return <Zap size={18} color="var(--purple)"/>
    return <CheckCircle size={18} color="var(--green)"/>
  }

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto' }}>
      <div className="page-header">
        <div className="page-eyebrow">Interactive Testing</div>
        <h1 className="page-title">GaaS Live Playground</h1>
        <p className="page-desc">Simulate how your LLM application will interact with the KavachX Governance Engine.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 2.5fr', gap: 24 }}>
        {/* Left: Configuration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <Card title="1. Select Application Context">
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
              Tell the engine what type of app you are building to apply domain-specific policies.
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {CONTEXTS.map(c => (
                <button 
                  key={c.id}
                  onClick={() => setSelectedContext(c.id)}
                  className={`btn btn-sm ${selectedContext === c.id ? 'active' : ''}`}
                  style={{ 
                    justifyContent: 'flex-start', textAlign: 'left', padding: '12px', 
                    background: selectedContext === c.id ? 'var(--accent-light)' : 'var(--bg-elevated)', 
                    border: '1px solid ' + (selectedContext === c.id ? 'var(--accent)' : 'var(--border)'),
                    transition: 'all 0.2s'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <c.icon size={16} color={selectedContext === c.id ? 'var(--accent)' : 'var(--text-muted)'} />
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: selectedContext === c.id ? 'var(--accent)' : 'var(--text)' }}>{c.label}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{c.desc}</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Reality Check">
            <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              <p><strong>Actual Real-World Use:</strong></p>
              <p>In a real deployment, you don't use this UI. You use the <strong>KavachX SDK</strong> or <strong>REST API</strong> from your Python/Node server.</p>
              <p style={{ marginTop: 8 }}>When your user typed something in your app, your server would send it to your Render URL for a "Traffic Light" decision before ever showing it to the LLM.</p>
            </div>
          </Card>
        </div>

        {/* Right: Interaction */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <Card title="2. Execute Manual Prompt">
            <form onSubmit={handleTestManual} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ position: 'relative' }}>
                <textarea 
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  placeholder="Type a prompt to test policy violations... (e.g. 'Show me my neighbors debt records')"
                  style={{ 
                    width: '100%', minHeight: 100, padding: '12px 14px', borderRadius: 10, 
                    border: '1px solid var(--border)', background: 'var(--bg-input)', 
                    color: 'var(--text)', fontSize: 14, resize: 'none', lineHeight: 1.5
                  }}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleTestManual(e)}
                />
                <div style={{ position: 'absolute', bottom: 10, right: 10, fontSize: 10, color: 'var(--text-muted)' }}>
                  Press Ctrl+Enter to send
                </div>
              </div>
              <button 
                type="submit" 
                className="btn btn-primary" 
                disabled={loading || !prompt.trim()}
                style={{ alignSelf: 'flex-end', gap: 8, padding: '10px 24px' }}
              >
                {loading ? <Settings size={16} className="spin" /> : <Terminal size={16} />}
                Inspect Payload
              </button>
            </form>
          </Card>

          {/* Engine Output */}
          <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden', minHeight: 300 }}>
            <div className="card-header" style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="card-title">KavachX Governance Console</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>MODE: GAAS_ENFORCEMENT</div>
              </div>
            </div>
            
            <div style={{ flex: 1, background: '#0a0c16', color: '#8b949e', padding: 18, fontFamily: 'var(--font-mono)', fontSize: 12, overflow: 'auto' }}>
              {!result && !loading && (
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
                  <Search size={40} style={{ marginBottom: 12 }} />
                  <div>System standby. Enter a prompt to begin governance analysis.</div>
                </div>
              )}
              
              {loading && (
                <div className="fade-in" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div className="dot dot-blue spin" />
                  <span>Evaluating context against {selectedContext.toUpperCase()} policy stack...</span>
                </div>
              )}

              {result && (
                <div className="fade-in">
                  <div style={{ color: '#58a6ff', marginBottom: 6 }}>{`// Incoming Traffic Detected`}</div>
                  <div style={{ color: '#e6edf3', marginBottom: 16, padding: '10px', background: '#ffffff05', borderRadius: 4 }}>
                    {result.input}
                  </div>
                  
                  <div style={{ color: getStatusColor(result.response.decision), fontWeight: 700, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10, fontSize: 14 }}>
                    {getStatusIcon(result.response.decision)}
                    {`GAAS DECISION: ${result.response.decision}`}
                  </div>

                  <div style={{ background: '#ffffff03', border: '1px solid #ffffff10', padding: 16, borderRadius: 8 }}>
                    {result.response.violations?.length > 0 ? (
                      result.response.violations.map((v, i) => (
                        <div key={i} style={{ marginBottom: 12 }}>
                          <div style={{ color: '#ff7b72', fontWeight: 600, fontSize: 13 }}>• Violation: {v.policy_name}</div>
                          <div style={{ color: '#8b949e', fontSize: 11, marginLeft: 14, marginTop: 4 }}>
                            <strong>Engine Reason:</strong> {v.message}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div style={{ color: '#3fb950', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <CheckCircle size={16} /> 
                        <span>Clean Transaction. No violations against the metadata signatures.</span>
                      </div>
                    )}
                  </div>

                  <div style={{ marginTop: 24 }}>
                    <div style={{ color: '#58a6ff', fontSize: 10, marginBottom: 8 }}>{`// Engine Metadata Analysis`}</div>
                    <pre style={{ fontSize: 10, color: '#484f58', background: '#000', padding: 10, borderRadius: 4 }}>
                      {JSON.stringify(result.response.risk_analysis || {}, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

