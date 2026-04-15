import { useState, useEffect, useCallback } from 'react'
import Card from '../components/shared/Card'
import { useAuth } from '../context/AuthContext'
import { settingsAPI, usersAPI, dashboardAPI } from '../utils/api'
import { Users, Shield, Server, Save, Check, RefreshCw, AlertCircle } from 'lucide-react'

const ROLE_COLORS = {
  super_admin: '#3b82f6',
  compliance_officer: '#10b981',
  ml_engineer: '#06b6d4',
  auditor: '#f59e0b',
  executive: '#8b5cf6',
}

const ROLE_CAPS = {
  super_admin: ['Manage Roles', 'Configure System', 'Read Events', 'Write Policies'],
  compliance_officer: ['Read Events', 'Write Policies', 'Read Models'],
  ml_engineer: ['Read Events', 'Register Models', 'Run Simulations'],
  auditor: ['Read Events', 'Export Logs'],
  executive: ['Read Events', 'Read Models'],
}

export default function SettingsPage() {
  const { user } = useAuth()
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)
  const [thresholds, setThresholds] = useState({
    risk_high: 0.75, risk_medium: 0.45, fairness_disparity: 0.20, confidence_low: 0.60
  })
  const [liveUsers, setLiveUsers] = useState([])
  const [usersLoading, setUsersLoading] = useState(true)
  const [nodeStats, setNodeStats] = useState(null)

  const loadData = useCallback(async () => {
    const [tRes, uRes, sRes] = await Promise.allSettled([
      settingsAPI.getThresholds(),
      usersAPI.list(),
      dashboardAPI.getStats(),
    ])
    if (tRes.status === 'fulfilled' && tRes.value?.data) setThresholds(tRes.value.data)
    if (uRes.status === 'fulfilled' && uRes.value?.data) {
      setLiveUsers(Array.isArray(uRes.value.data) ? uRes.value.data : uRes.value.data.users || [])
    }
    if (sRes.status === 'fulfilled' && sRes.value?.data) setNodeStats(sRes.value.data)
    setUsersLoading(false)
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const handleSave = async () => {
    setSaving(true)
    try {
      await settingsAPI.updateThresholds(thresholds)
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch (e) {
      console.error(e)
    } finally {
      setSaving(false)
    }
  }

  const displayUsers = liveUsers.length > 0 ? liveUsers : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Row 1: Thresholds + Node Details */}
      <div className="grid-2" style={{ gap: 16 }}>

        {/* Governance Thresholds */}
        <Card title="Governance Thresholds" action={<Shield size={14} color="var(--accent)" />}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="alert alert-info" style={{ marginBottom: 0, padding: '10px 14px' }}>
              <div style={{ fontSize: 11, lineHeight: 1.5 }}>
                <strong>Live effect:</strong> These thresholds control High/Medium/Low risk
                labels on the Dashboard and determine when a request is blocked vs flagged
                for review in the governance engine.
              </div>
            </div>

            {[
              { key: 'risk_high', label: 'High Risk Threshold', desc: 'Requests scoring above this are blocked' },
              { key: 'risk_medium', label: 'Medium Risk Threshold', desc: 'Requests above this trigger a policy review' },
              { key: 'fairness_disparity', label: 'Fairness Disparity Max', desc: 'Max allowed gap across demographic groups' },
              { key: 'confidence_low', label: 'Low Confidence Threshold', desc: 'Predictions below this are flagged as uncertain' },
            ].map(f => (
              <div key={f.key}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>{f.label}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>{f.desc}</div>
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, color: 'var(--accent)', flexShrink: 0, marginLeft: 8 }}>
                    {(thresholds[f.key] * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.01"
                  value={thresholds[f.key]}
                  onChange={e => setThresholds(t => ({ ...t, [f.key]: parseFloat(e.target.value) }))}
                  style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer', marginTop: 4 }}
                />
              </div>
            ))}

            <button
              className={`btn btn-sm ${saved ? 'btn-secondary' : 'btn-primary'}`}
              onClick={handleSave}
              disabled={saving}
              style={{ alignSelf: 'flex-end' }}
            >
              {saved ? <><Check size={13} /> Applied Globally</> : saving ? <><RefreshCw size={13} style={{ animation: 'spin 1s linear infinite' }} /> Saving…</> : <><Save size={13} /> Save &amp; Update Engine</>}
            </button>
          </div>
        </Card>

        {/* Kavach Node Details — live stats */}
        <Card title="Kavach Node Details" action={<Server size={14} color="var(--accent)" />}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              ['Deployment ID', 'KAVACH-NODE-IN-001'],
              ['Engine Version', '2.0.0-PROD'],
              ['Jurisdiction', 'India (MeitY)'],
              ['Compliance Scope', 'DPDPA, EU AI Act'],
              ['Total Inferences', nodeStats ? nodeStats.total_inferences.toLocaleString() : '—'],
              ['Active Models', nodeStats ? nodeStats.active_models : '—'],
              ['Avg Risk Score', nodeStats ? (nodeStats.avg_risk_score * 100).toFixed(1) + '%' : '—'],
              ['Pass Rate', nodeStats ? (nodeStats.pass_rate * 100).toFixed(1) + '%' : '—'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>{k}</span>
                <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text)', fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Row 2: RBAC & User Management — live from API */}
      <Card title="RBAC & Account Management" action={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {usersLoading && <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite', color: 'var(--text-muted)' }} />}
          <Users size={14} color="var(--accent)" />
        </div>
      }>
        {displayUsers.length === 0 && !usersLoading ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '16px 0', color: 'var(--text-muted)', fontSize: 13 }}>
            <AlertCircle size={14} />
            No user records returned. Verify the /users API is reachable with your current role.
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th style={{ minWidth: 80 }}>Org</th>
                  <th>Capabilities</th>
                </tr>
              </thead>
              <tbody>
                {displayUsers.map(u => {
                  const color = ROLE_COLORS[u.role] || '#64748b'
                  const caps = ROLE_CAPS[u.role] || []
                  const initials = (u.name || u.email || '??').split(' ').map(w => w[0]).slice(0, 2).join('').toUpperCase()
                  return (
                    <tr key={u.id || u.email}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: 26, height: 26, borderRadius: 6, background: color + '20', border: '1px solid ' + color + '40', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, fontWeight: 700, color, flexShrink: 0 }}>
                            {initials}
                          </div>
                          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)', whiteSpace: 'nowrap' }}>{u.name || '—'}</span>
                        </div>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{u.email}</td>
                      <td>
                        <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 99, background: color + '20', color, border: '1px solid ' + color + '30', fontWeight: 600, whiteSpace: 'nowrap' }}>
                          {u.role?.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td style={{ fontSize: 12 }}>{u.org || '—'}</td>
                      <td style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        {caps.slice(0, 3).join(', ')}{caps.length > 3 ? ` +${caps.length - 3}` : ''}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Row 3: API Access — with margin-top via gap on parent */}
      <Card title="Kavach API Access & Connectivity">
        <div className="alert alert-info" style={{ marginBottom: 16 }}>
          <span style={{ fontSize: 12 }}>
            These keys allow external LLMs and ML models to connect to the KavachX
            Governance Layer. Rotate keys quarterly or after any suspected exposure.
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            { name: 'Production Kavach Core Key', key: 'kavachx-demo-key', status: 'active', role: 'ML Engineer (System)', scope: 'governance:evaluate, models:read' },
          ].map(k => (
            <div
              key={k.name}
              style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, padding: '12px 16px', background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
            >
              <div style={{ flex: 1, minWidth: 220 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{k.name}</div>
                <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginTop: 6 }}>
                  <code style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--accent)', background: 'var(--accent-light)', padding: '2px 6px', borderRadius: 4 }}>{k.key}</code>
                  <button
                    onClick={() => navigator.clipboard.writeText(k.key)}
                    style={{ fontSize: 11, color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', textDecoration: 'underline', padding: 0 }}>
                    Copy
                  </button>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>Scope: {k.scope}</div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <span className="badge badge-active">{k.status}</span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>Access: {k.role}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

    </div>
  )
}
