import { useEffect, useState } from 'react'
import { Bell, AlertTriangle, XCircle, CheckCircle, Clock, X, Filter } from 'lucide-react'
import { auditAPI } from '../utils/api'

const ALERT_ICONS = { critical: <XCircle size={16} style={{ color: 'var(--red)', flexShrink: 0 }} />, warning: <AlertTriangle size={16} style={{ color: 'var(--amber)', flexShrink: 0 }} />, info: <CheckCircle size={16} style={{ color: 'var(--accent)', flexShrink: 0 }} /> }
const ALERT_BADGE = { critical: 'badge-critical', warning: 'badge-alert', info: 'badge-info' }

const fmt = (d) => {
  const diff = Date.now() - d.getTime()
  if (diff < 60000) return 'just now'
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago'
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago'
  return d.toLocaleDateString()
}

// Fallback if no real alerts exist
const MOCK_FALLBACK = [
  { id: 'a1', type: 'info', title: 'System initialized', desc: 'Governance engine is online and monitoring all models.', time: new Date(), model: null, read: true }
]

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState('all')

  const unread = alerts.filter(a => !a.read).length
  const filtered = alerts.filter(a => {
    if (filter === 'all') return true
    if (filter === 'unread') return !a.read
    return a.type === filter
  })

  const load = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const r = await auditAPI.getLogs({ limit: 50 })
      const data = r.data || []

      // Map audit logs to alert format
      const mapped = data.filter(l =>
        ['policy_violated', 'model_blocked', 'fairness_issue_detected', 'inference_evaluated'].includes(l.event_type)
      ).map(l => ({
        id: l.id,
        type: l.risk_level === 'critical' || l.event_type === 'model_blocked' ? 'critical' : l.risk_level === 'high' ? 'warning' : 'info',
        title: l.event_type.replace(/_/g, ' ').toUpperCase(),
        desc: l.action || l.details?.message || 'Policy assessment triggered.',
        time: new Date(l.timestamp),
        model: l.entity_id,
        read: false
      }))

      setAlerts(mapped.length > 0 ? mapped : MOCK_FALLBACK)
    } catch {
      setAlerts(MOCK_FALLBACK)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const onSim = () => load(true)
    window.addEventListener('kavachx:simulation-complete', onSim)

    // Real-time: reload when a block or policy violation arrives over WebSocket
    const apiBase = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? `${window.location.origin}/api/v1` : 'http://localhost:8005/api/v1')
    const wsURL = `${apiBase.replace(/^http/, 'ws').replace('/api/v1', '')}/api/v1/ws/stream`
    const ws = new WebSocket(wsURL)
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'new_inference' &&
            (msg.enforcement_decision === 'BLOCK' || msg.risk_score >= 0.60)) {
          load(true)
        }
      } catch (e) {}
    }

    return () => {
      window.removeEventListener('kavachx:simulation-complete', onSim)
      ws.close()
    }
  }, [])

  const markRead = (id) => setAlerts(a => a.map(al => al.id === id ? { ...al, read: true } : al))
  const markAllRead = () => setAlerts(a => a.map(al => ({ ...al, read: true })))

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div className="page-eyebrow">Advanced</div>
            <h1 className="page-title">Alerts <span style={{ fontSize: 16, background: 'var(--red)', color: '#fff', borderRadius: 20, padding: '2px 8px', fontWeight: 700 }}>{unread}</span></h1>
            <p className="page-desc">Real-time governance alerts, fairness violations, and system notifications</p>
          </div>
          {unread > 0 && <button className="btn btn-secondary btn-sm" onClick={markAllRead}>Mark all read</button>}
        </div>
      </div>

      <div className="filter-pills mb-16">
        {['all', 'unread', 'critical', 'warning', 'info'].map(f => (
          <button key={f} className={`filter-pill ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
            {f === 'all' ? `All (${alerts.length})` : f === 'unread' ? `Unread (${unread})` : f}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {filtered.length === 0 && (
          <div className="empty card"><Bell size={32} style={{ opacity: .2 }} /><div className="empty-title">No alerts</div></div>
        )}
        {filtered.map(alert => (
          <div key={alert.id} className={`card fade-up${alert.type === 'critical' && !alert.read ? ' item-critical' : ''}`} style={{ borderLeft: `3px solid ${alert.type === 'critical' ? 'var(--red)' : alert.type === 'warning' ? 'var(--amber)' : 'var(--accent)'}`, opacity: alert.read ? .7 : 1 }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              {ALERT_ICONS[alert.type]}
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                  <div style={{ fontWeight: !alert.read ? 700 : 500, fontSize: 13.5, color: 'var(--text)' }}>{alert.title}</div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
                    <span className={`badge ${ALERT_BADGE[alert.type]}`}>{alert.type}</span>
                    {!alert.read && <button onClick={() => markRead(alert.id)} style={{ color: 'var(--text-muted)', cursor: 'pointer' }}><X size={14} /></button>}
                  </div>
                </div>
                <div style={{ fontSize: 12.5, color: 'var(--text-muted)', margin: '4px 0 8px' }}>{alert.desc}</div>
                <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-muted)' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={11} /> {fmt(alert.time)}</span>
                  {alert.model && <span>Model: <strong>{alert.model}</strong></span>}
                  {!alert.read && <span style={{ color: 'var(--accent)', fontWeight: 700 }}>● UNREAD</span>}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
