import { useEffect, useState, useCallback } from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { dashboardAPI, auditAPI } from '../utils/api'
import { Scale, FileCheck, AlertTriangle, CheckCircle2, Download, RefreshCw, AlertOctagon } from 'lucide-react'
import { getRiskBadge, timeAgo } from '../utils/helpers'

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
function useChartColors() {
  const [c, setC] = useState({})
  useEffect(() => {
    const read = () => setC({ accent: cssVar('--accent') || '#4f46e5', border: cssVar('--border') || '#e0e2ed' })
    read()
    const obs = new MutationObserver(read)
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return c
}

// ISO 42001 is not yet derivable from audit data — kept as a static benchmark target.
const ISO_STATIC = { framework: 'ISO 42001', score: 55, status: 'gap' }

const INITIAL_STATS = { total_inferences: 0, policy_violations_today: 0, active_models: 0, fairness_issues_detected: 0 }

const statusColor = (s) => ({ compliant: 'var(--green)', partial: 'var(--amber)', gap: 'var(--red)', non_compliant: 'var(--red)' }[s] || 'var(--text-muted)')
const statusBadge = (s) => ({ compliant: 'badge-pass', partial: 'badge-alert', gap: 'badge-block', non_compliant: 'badge-block' }[s] || 'badge-muted')
const statusLabel = (s) => ({ compliant: 'compliant', partial: 'partial', gap: 'gap', non_compliant: 'non-compliant' }[s] || s)

// Color-coded event type badges — violations and blocks stand out immediately
const EVENT_BADGE = {
  policy_violated:         'badge-block',
  model_blocked:           'badge-block',
  fairness_issue_detected: 'badge-alert',
  inference_evaluated:     'badge-info',
  model_registered:        'badge-pass',
  policy_created:          'badge-pass',
}
const eventBadge = (et) => EVENT_BADGE[et] || 'badge-muted'

const RISK_FILTERS = ['all', 'critical', 'high', 'medium', 'low']

export default function ComplianceDashboard() {
  const [stats, setStats] = useState(INITIAL_STATS)
  const [frameworks, setFrameworks] = useState([])
  const [violations, setViolations] = useState([])
  const [loading, setLoading] = useState(true)
  const [riskFilter, setRiskFilter] = useState('all')
  const col = useChartColors()

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const [s, v, f] = await Promise.allSettled([
        dashboardAPI.getStats(),
        auditAPI.getLogs({ limit: 20 }),
        dashboardAPI.getComplianceSummary(),
      ])
      if (s.status === 'fulfilled' && s.value?.data) setStats(s.value.data)
      if (v.status === 'fulfilled' && v.value?.data) setViolations(v.value.data)
      if (f.status === 'fulfilled' && f.value?.data) {
        const live = Array.isArray(f.value.data) ? f.value.data : []
        setFrameworks([...live, ISO_STATIC])
      }
    } catch (err) {
      console.error('Failed to load compliance data:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()

    const apiBase = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? `${window.location.origin}/api/v1` : 'http://localhost:8002/api/v1')
    const wsBase = apiBase.replace(/^http/, 'ws').replace('/api/v1', '')
    const wsURL = `${wsBase}/api/v1/ws/stream`
    let ws
    try {
      ws = new WebSocket(wsURL)
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'new_inference') load(true)
        } catch (e) {}
      }
    } catch (e) {}

    const handleRefresh = () => load(true)
    window.addEventListener('kavachx:simulation-complete', handleRefresh)

    return () => {
      window.removeEventListener('kavachx:simulation-complete', handleRefresh)
      if (ws) ws.close()
    }
  }, [load])

  const exportComplianceReport = () => {
    const lines = [
      'KavachX Compliance Report',
      `Generated: ${new Date().toLocaleString()}`,
      '',
      'FRAMEWORK SCORES',
      ...frameworks.map(f => `${f.framework || f.label},${f.score}%,${f.status}`),
      '',
      'SUMMARY STATISTICS',
      `Total Inferences,${stats?.total_inferences ?? 'N/A'}`,
      `Policy Violations Today,${stats?.policy_violations_today ?? 'N/A'}`,
      `Active Models,${stats?.active_models ?? 'N/A'}`,
      `Fairness Issues,${stats?.fairness_issues_detected ?? 'N/A'}`,
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `kavachx-compliance-report-${Date.now()}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  const exportViolations = () => {
    const rows = [['Timestamp', 'Event Type', 'Entity', 'Actor', 'Risk Level'].join(',')]
    violations.forEach(v => rows.push([
      new Date(v.timestamp).toISOString(), v.event_type, v.entity_id || '', v.actor || '', v.risk_level || ''
    ].join(',')))
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `kavachx-violations-${Date.now()}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  const radarData = frameworks.map(f => ({ subject: f.framework || f.label, score: f.score }))
  const compliant = frameworks.filter(f => f.status === 'compliant').length
  const partial = frameworks.filter(f => f.status === 'partial').length
  const gaps = frameworks.filter(f => f.status === 'gap' || f.status === 'non_compliant').length
  const criticalCount = violations.filter(v => v.risk_level === 'critical').length

  const displayedViolations = riskFilter === 'all'
    ? violations
    : violations.filter(v => v.risk_level === riskFilter)

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
          <div>
            <div className="page-eyebrow">Compliance Officer View</div>
            <h1 className="page-title">Compliance Dashboard</h1>
            <p className="page-desc">Regulatory compliance posture, policy coverage, and audit readiness</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {loading && <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite', color: 'var(--text-muted)' }} />}
            <button className="btn btn-secondary btn-sm" onClick={exportComplianceReport}>
              <Download size={14} /> Export Report
            </button>
          </div>
        </div>
      </div>

      <div className="stats-row">
        {[
          { label: 'Compliant Frameworks', value: loading ? '—' : compliant, icon: CheckCircle2, color: 'var(--green)', bg: 'var(--green-light)' },
          { label: 'Partial Compliance', value: loading ? '—' : partial, icon: AlertTriangle, color: 'var(--amber)', bg: 'var(--amber-light)' },
          { label: 'Coverage Gaps', value: loading ? '—' : gaps, icon: Scale, color: 'var(--red)', bg: 'var(--red-light)' },
          { label: 'Critical Events', value: loading ? '—' : criticalCount, icon: AlertOctagon, color: criticalCount > 0 ? 'var(--red)' : 'var(--text-muted)', bg: criticalCount > 0 ? 'var(--red-light)' : 'var(--bg-elevated)' },
        ].map(({ label, value, icon: Icon, color, bg }) => (
          <div key={label} className="stat-card" style={{ '--stat-color': color, '--stat-bg': bg }}>
            <div className="stat-icon"><Icon size={18} /></div>
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      <div className="grid-2 mb-20">
        {/* Radar */}
        <div className="card">
          <div className="card-header"><span className="card-title">Compliance Radar</span></div>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '8px 0' }}>
              <div className="skeleton skeleton-card" style={{ height: 180 }} />
            </div>
          ) : frameworks.length === 0 ? (
            <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              No framework data yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData}>
                <PolarGrid stroke={col.border || '#e0e2ed'} />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <Radar name="Score" dataKey="score" stroke={col.accent || '#4f46e5'} fill={col.accent || '#4f46e5'} fillOpacity={0.15} strokeWidth={2} />
                <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
              </RadarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Framework Scores */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Framework Scores</span>
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Derived from last 30 days of audit data</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {loading ? (
              [1, 2, 3, 4].map(i => (
                <div key={i}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <div className="skeleton skeleton-text" style={{ width: '40%', height: 12 }} />
                    <div className="skeleton skeleton-text" style={{ width: '20%', height: 12 }} />
                  </div>
                  <div className="skeleton" style={{ height: 5, borderRadius: 3 }} />
                </div>
              ))
            ) : frameworks.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '8px 0' }}>No data</div>
            ) : (
              frameworks.map(f => {
                const name = f.framework || f.label
                const sc = f.score
                const color = statusColor(f.status)
                return (
                  <div key={name}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>{name}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 12, fontWeight: 700, color }}>{sc}%</span>
                        <span className={`badge ${statusBadge(f.status)}`}>{statusLabel(f.status)}</span>
                      </div>
                    </div>
                    <div className="risk-bar">
                      <div className="risk-fill" style={{ width: sc + '%', background: color }} />
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>

      {/* Recent audit events */}
      <div className="card">
        <div className="card-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="card-title">Recent Audit Events</span>
            {criticalCount > 0 && (
              <span className="badge badge-critical">
                <AlertOctagon size={10} />
                {criticalCount} critical
              </span>
            )}
          </div>
          <button className="btn btn-ghost btn-sm" onClick={exportViolations}><Download size={14} /> Export</button>
        </div>

        {/* Risk level filter pills */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 14, flexWrap: 'wrap' }}>
          {RISK_FILTERS.map(level => {
            const count = level === 'all' ? violations.length : violations.filter(v => v.risk_level === level).length
            return (
              <button
                key={level}
                className={`filter-pill ${riskFilter === level ? 'active' : ''}`}
                onClick={() => setRiskFilter(level)}
                style={{ display: 'flex', alignItems: 'center', gap: 5 }}
              >
                {level === 'all' ? 'All' : level.charAt(0).toUpperCase() + level.slice(1)}
                {count > 0 && (
                  <span style={{
                    fontSize: 9, fontWeight: 700,
                    background: riskFilter === level ? 'rgba(255,255,255,0.25)' : 'var(--bg-elevated)',
                    color: riskFilter === level ? '#fff' : 'var(--text-muted)',
                    padding: '1px 5px', borderRadius: 8, lineHeight: 1.4,
                  }}>{count}</span>
                )}
              </button>
            )
          })}
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Event Type</th>
                <th>Actor</th>
                <th>Entity</th>
                <th>Risk Level</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                [1, 2, 3, 4, 5].map(i => (
                  <tr key={i}>
                    {[1, 2, 3, 4, 5].map(j => (
                      <td key={j}><div className="skeleton skeleton-text" style={{ height: 14, width: j === 1 ? '70%' : j === 4 ? '40%' : '80%' }} /></td>
                    ))}
                  </tr>
                ))
              ) : displayedViolations.length === 0 ? (
                <tr>
                  <td colSpan={5}>
                    <div className="empty" style={{ padding: '28px 0' }}>
                      <FileCheck size={28} style={{ opacity: 0.2, marginBottom: 6 }} />
                      <div className="empty-title">No events</div>
                      <div className="empty-desc">
                        {riskFilter === 'all' ? 'No audit events recorded yet.' : `No ${riskFilter} events in recent logs.`}
                      </div>
                    </div>
                  </td>
                </tr>
              ) : (
                displayedViolations.map(v => (
                  <tr key={v.id} className={v.risk_level === 'critical' ? 'row-critical' : ''}>
                    <td
                      className="font-mono"
                      style={{ fontSize: 11, whiteSpace: 'nowrap', color: 'var(--text-dim)' }}
                      title={new Date(v.timestamp).toLocaleString()}
                    >
                      {timeAgo(v.timestamp)}
                    </td>
                    <td>
                      <span className={`badge ${eventBadge(v.event_type)}`}>
                        {v.event_type?.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td style={{ fontSize: 12 }}>{v.actor || '—'}</td>
                    <td style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {v.entity_id ? v.entity_id.substring(0, 12) + '…' : '—'}
                    </td>
                    <td>
                      {v.risk_level ? (
                        <span className={`badge ${getRiskBadge(v.risk_level)}`}>
                          {v.risk_level}
                        </span>
                      ) : '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
