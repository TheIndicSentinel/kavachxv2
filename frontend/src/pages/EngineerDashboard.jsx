import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { modelsAPI, governanceAPI } from '../utils/api'
import { Database, Cpu, CheckCircle, AlertCircle } from 'lucide-react'

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

const DEFAULT_COLORS = {
  accent: '#4f46e5',
  border: '#e0e2ed',
  bgCard: '#ffffff',
}

function useChartColors() {
  const [c, setC] = useState(DEFAULT_COLORS)
  useEffect(() => {
    const read = () => {
      const resolved = {
        accent: cssVar('--accent') || DEFAULT_COLORS.accent,
        border: cssVar('--border') || DEFAULT_COLORS.border,
        bgCard: cssVar('--bg-card') || DEFAULT_COLORS.bgCard,
      }
      setC(resolved)
    }
    read()
    const obs = new MutationObserver(read)
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return c
}

// Decision → badge class AND color (single source of truth)
const DECISION_CONFIG = {
  PASS:         { badge: 'badge-pass',   color: 'var(--green)',  label: 'ALLOW' },
  ALERT:        { badge: 'badge-alert',  color: 'var(--red)',  label: 'ALERT' },
  HUMAN_REVIEW: { badge: 'badge-review', color: 'var(--orange)', label: 'REVIEW' },
  BLOCK:        { badge: 'badge-block',  color: 'var(--red)',    label: 'BLOCK' },
}

const decisionBadge = (d) => {
  const cfg = DECISION_CONFIG[(d || '').toUpperCase()] || { badge: 'badge-muted', label: d }
  return <span className={`badge ${cfg.badge}`}>{cfg.label}</span>
}

const decisionColor = (d) => {
  const cfg = DECISION_CONFIG[(d || '').toUpperCase()]
  return cfg ? cfg.color : 'var(--green)'
}

const TOOLTIP_STYLE = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  borderRadius: 8,
  fontSize: 12,
}

export default function EngineerDashboard() {
  const [models, setModels] = useState([])
  const [inferences, setInferences] = useState([])
  const [loading, setLoading] = useState(true)
  const col = useChartColors()

  useEffect(() => {
    const load = async () => {
      try {
        const [m, inf] = await Promise.all([
          modelsAPI.list(),
          governanceAPI.getInferences({ limit: 30 }),
        ])
        if (m.data) setModels(m.data)
        if (inf.data) setInferences(inf.data)
      } catch (err) {
        console.error("Failed to load engineer dashboard data:", err)
      } finally {
        setLoading(false)
      }
    }
    load()
    const iv = setInterval(load, 15000)
    
    // Live WebSocket connection
    const apiBase = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? `${window.location.origin}/api/v1` : 'http://localhost:8005/api/v1')
    const wsBase = apiBase.replace(/^http/, 'ws').replace('/api/v1', '')
    const wsURL = `${wsBase}/api/v1/ws/stream`
    const ws = new WebSocket(wsURL)
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'new_inference') load()
      } catch(e) {}
    }

    const onSim = () => load()
    window.addEventListener('kavachx:simulation-complete', onSim)
    
    return () => { 
      clearInterval(iv)
      window.removeEventListener('kavachx:simulation-complete', onSim)
      ws.close()
    }
  }, [])

  const displayModels = models
  const displayInferences = inferences

  /* Build deterministic bar bins from inference data */
  const riskBins = [0, 0, 0, 0, 0]
  displayInferences.forEach(inf => {
    const bin = Math.min(4, Math.floor((inf.risk_score || 0) * 5))
    riskBins[bin]++
  })
  const barData = ['0–20', '20–40', '40–60', '60–80', '80–100'].map((label, i) => ({
    label,
    count: riskBins[i],
  }))

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">ML Engineer View</div>
        <h1 className="page-title">System Monitor</h1>
        <p className="page-desc">Model performance, inference pipeline and technical governance metrics</p>
      </div>

      <div className="stats-row">
        {[
          { label: 'Registered Models', value: displayModels.length, icon: Database, color: 'var(--accent)', bg: 'var(--accent-light)' },
          { label: 'Active Models', value: displayModels.filter(m => m.status === 'active').length, icon: CheckCircle, color: 'var(--green)', bg: 'var(--green-light)' },
          { label: 'Suspended', value: displayModels.filter(m => m.status === 'suspended').length, icon: AlertCircle, color: 'var(--red)', bg: 'var(--red-light)' },
          { label: 'Inferences (recent)', value: displayInferences.length, icon: Cpu, color: 'var(--purple)', bg: 'var(--purple-light)' },
        ].map(({ label, value, icon: Icon, color, bg }) => (
          <div key={label} className="stat-card" style={{ '--stat-color': color, '--stat-bg': bg }}>
            <div className="stat-icon"><Icon size={18} /></div>
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      <div className="grid-2 mb-20">
        {/* ── Risk Score Distribution Bar Chart ── */}
        <div className="card">
          <div className="card-header"><span className="card-title">Risk Score Distribution</span></div>
          {col.accent ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={col.border} />
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} allowDecimals={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="count" fill={col.accent} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              Loading…
            </div>
          )}
        </div>

        {/* ── Registered Models ── */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Registered Models</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{displayModels.length} total</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 220, overflowY: 'auto' }}>
            {displayModels.length === 0 && (
              <div className="empty"><div className="empty-title">No models registered</div></div>
            )}
            {displayModels.map(m => (
              <div key={m.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 8 }}>
                <Database size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.name}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{m.version} · {m.model_type}</div>
                </div>
                <span className={`badge ${m.status === 'active' ? 'badge-active' : m.status === 'suspended' ? 'badge-block' : 'badge-muted'}`}>{m.status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Recent Inference Events ── */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Inference Events</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div className="dot dot-green" />
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>LIVE</span>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>Model</th>
                <th>Platform</th>
                <th>Decision</th>
                <th>Policy Triggered</th>
                <th>Reason</th>
                <th>Risk</th>
                <th>Confidence</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {displayInferences.length === 0 && (
                <tr><td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>No inferences yet. Use the browser extension or Simulate to generate data.</td></tr>
              )}
              {displayInferences.slice(0, 20).map(inf => {
                const prompt = inf.input_data?.prompt || inf.input_data?.text || '—'
                const reason = inf.explanation?.reason || '—'
                const platform = inf.input_data?.platform || inf.context_metadata?.platform || 'Universal'
                const policyTriggered = inf.explanation?.policy_triggered || (inf.policy_violations?.[0]?.policy_name) || (reason === 'No policy violation detected.' ? '—' : 'System Rule')
                
                return (
                  <tr key={inf.id}>
                    <td style={{ fontSize: 11, color: 'var(--text)', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={prompt}>
                      {prompt.length > 50 ? prompt.substring(0, 50) + '…' : prompt}
                    </td>
                    <td style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>{inf.model_id}</td>
                    <td><span className="badge badge-info" style={{ fontSize: 9 }}>{platform}</span></td>
                    <td>{decisionBadge(inf.enforcement_decision)}</td>
                    <td style={{ fontSize: 11, color: 'var(--text-dim)', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={policyTriggered}>
                      {policyTriggered}
                    </td>
                    <td style={{ fontSize: 11, color: 'var(--text-dim)', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={reason}>
                      {reason}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="risk-bar" style={{ width: 50 }}>
                          <div className="risk-fill" style={{
                            width: (inf.risk_score || 0) * 100 + '%',
                            background: decisionColor(inf.enforcement_decision),
                          }} />
                        </div>
                        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                          {((inf.risk_score || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="font-mono" style={{ fontSize: 11 }}>{((inf.confidence || 0) * 100).toFixed(1)}%</td>
                    <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {inf.timestamp ? new Date(inf.timestamp).toLocaleTimeString() : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
