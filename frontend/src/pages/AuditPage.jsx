import { useEffect, useState, useCallback } from 'react'
import { auditAPI } from '../utils/api'
import { Download, Search, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react'
import { getRiskBadge } from '../utils/helpers'

const EVENT_TYPES = ['all', 'inference_evaluated', 'policy_violated', 'model_blocked', 'model_registered', 'policy_created']
const PAGE_SIZE = 15

function Pagination({ page, total, pageSize, onChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  if (totalPages <= 1) return null
  const pages = []
  const start = Math.max(1, page - 2)
  const end = Math.min(totalPages, page + 2)
  for (let i = start; i <= end; i++) pages.push(i)

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 0 2px', borderTop: '1px solid var(--border)' }}>
      <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
        Showing {Math.min((page - 1) * pageSize + 1, total)}–{Math.min(page * pageSize, total)} of {total}
      </span>
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <button className="btn btn-secondary btn-xs" onClick={() => onChange(page - 1)} disabled={page === 1}>
          <ChevronLeft size={13} />
        </button>
        {start > 1 && <><button className="btn btn-secondary btn-xs" onClick={() => onChange(1)}>1</button><span style={{ color: 'var(--text-muted)', fontSize: 11 }}>…</span></>}
        {pages.map(p => (
          <button key={p} className={`btn btn-xs ${p === page ? 'btn-primary' : 'btn-secondary'}`} onClick={() => onChange(p)}>{p}</button>
        ))}
        {end < totalPages && <><span style={{ color: 'var(--text-muted)', fontSize: 11 }}>…</span><button className="btn btn-secondary btn-xs" onClick={() => onChange(totalPages)}>{totalPages}</button></>}
        <button className="btn btn-secondary btn-xs" onClick={() => onChange(page + 1)} disabled={page === totalPages}>
          <ChevronRight size={13} />
        </button>
      </div>
    </div>
  )
}

const DECISION_BADGE = {
  PASS: 'badge-pass', ALLOW: 'badge-pass',
  ALERT: 'badge-alert',
  HUMAN_REVIEW: 'badge-review', REVIEW: 'badge-review',
  BLOCK: 'badge-block',
}

const EVENT_BADGE = {
  policy_violated:         'badge-block',
  model_blocked:           'badge-block',
  fairness_issue_detected: 'badge-alert',
  inference_evaluated:     'badge-info',
  model_registered:        'badge-pass',
  policy_created:          'badge-pass',
}
const eventBadge = (et) => EVENT_BADGE[et] || 'badge-muted'

export default function AuditPage() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [selectedLog, setSelectedLog] = useState(null)

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true)
    try {
      const params = { limit: 200 }
      if (filter !== 'all') params.event_type = filter
      const r = await auditAPI.getLogs(params)
      setLogs(Array.isArray(r.data) && r.data.length > 0 ? r.data : [])
    } catch { setLogs([]) }
    finally { if (showSpinner) setLoading(false); else setLoading(false) }
  }, [filter])

  useEffect(() => { load(true) }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { load(true); setPage(1) }, [filter]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const handler = () => { load(false) }
    window.addEventListener('kavachx:simulation-complete', handler)
    return () => window.removeEventListener('kavachx:simulation-complete', handler)
  }, [load])

  useEffect(() => { setPage(1) }, [search])

  const filtered = search
    ? logs.filter(l =>
      l.event_type?.toLowerCase().includes(search.toLowerCase()) ||
      l.actor?.toLowerCase().includes(search.toLowerCase()) ||
      l.entity_id?.toLowerCase().includes(search.toLowerCase()) ||
      l.details?.prompt?.toLowerCase().includes(search.toLowerCase()) ||
      l.details?.reason?.toLowerCase().includes(search.toLowerCase())
    )
    : logs

  const totalFiltered = filtered.length
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  const riskBadge = getRiskBadge

    const exportCSV = () => {
    const rows = [['Timestamp', 'Session ID', 'Event Type', 'Decision', 'Risk Score', 'Risk Level', 'Prompt', 'Policy Triggered', 'Reason', 'Model', 'Platform'].join(',')]
    filtered.forEach(l => rows.push([
      new Date(l.timestamp).toISOString(),
      l.details?.session_id || '',
      l.event_type,
      l.details?.decision || l.action || '',
      l.details?.risk_score || '',
      l.risk_level || '',
      `"${(l.details?.prompt || '').replace(/"/g, "'")}"`,
      l.details?.policy_triggered || '',
      `"${(l.details?.reason || '').replace(/"/g, "'")}"`,
      l.actor || '',
      l.details?.platform || ''
    ].join(',')))
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `kavachx-audit-${Date.now()}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div className="page-eyebrow">Governance</div>
            <h1 className="page-title">Audit Logs</h1>
            <p className="page-desc">Immutable record of all governance events, policy violations, and model actions</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw size={13} /> Refresh</button>
            <button className="btn btn-secondary btn-sm" onClick={exportCSV}><Download size={13} /> Export CSV</button>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input className="form-input" style={{ paddingLeft: 32 }} placeholder="Search by prompt, reason, actor…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="filter-pills">
          {EVENT_TYPES.map(t => (
            <button key={t} className={`filter-pill ${filter === t ? 'active' : ''}`} onClick={() => setFilter(t)}>
              {t === 'all' ? 'All' : t.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Audit Trail</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{totalFiltered} events</span>
        </div>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}><div className="spinner" /></div>
        ) : filtered.length === 0 ? (
          <div className="empty">
            <div className="empty-title">No audit events</div>
            <div className="empty-desc">Run inferences or create policies to generate audit events.</div>
          </div>
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Session ID</th>
                    <th>Event</th>
                    <th>Prompt</th>
                    <th>Decision</th>
                    <th>Risk</th>
                    <th>Policy Triggered</th>
                    <th>Reason</th>
                    <th>Platform</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {paginated.map(l => {
                    const decision = l.details?.decision || l.action || '—'
                    const prompt = l.details?.prompt || '—'
                    const reason = l.details?.reason || '—'
                    const policy = l.details?.policy_triggered || '—'
                    const platform = l.details?.platform || '—'
                    const sessionId = l.details?.session_id || '—'
                    const riskScore = l.details?.risk_score != null ? (l.details.risk_score * 100).toFixed(0) + '%' : '—'
                    
                    return (
                      <tr
                        key={l.id}
                        onClick={() => setSelectedLog(l)}
                        style={{ cursor: 'pointer', transition: 'background-color 0.15s ease' }}
                        className={`hover-row${l.risk_level === 'critical' ? ' row-critical' : ''}`}
                      >
                        <td className="font-mono" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>{new Date(l.timestamp).toLocaleString()}</td>
                        <td className="font-mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{sessionId.substring(0,8)}...</td>
                        <td><span className={`badge ${eventBadge(l.event_type)}`}>{l.event_type?.replace(/_/g, ' ')}</span></td>
                        <td style={{ fontSize: 11, maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={prompt}>
                          {prompt.length > 40 ? prompt.substring(0, 40) + '…' : prompt}
                        </td>
                        <td>
                          {decision !== '—' ? (
                            <span className={`badge ${DECISION_BADGE[decision.toUpperCase()] || 'badge-muted'}`}>{decision}</span>
                          ) : '—'}
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            {l.risk_level ? <span className={`badge ${riskBadge(l.risk_level)}`}>{l.risk_level}</span> : null}
                            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{riskScore}</span>
                          </div>
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--text-dim)', maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={policy}>
                          {policy}
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={reason}>
                          {reason}
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--text-dim)' }}>{platform}</td>
                        <td style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>{l.actor || '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <Pagination page={page} total={totalFiltered} pageSize={PAGE_SIZE} onChange={setPage} />
          </>
        )}
      </div>

      {/* --- SIDE DRAWER FOR AUDIT DETAILS --- */}
      {selectedLog && (
        <>
          <div
            style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(2, 6, 23, 0.45)', zIndex: 1000, backdropFilter: 'blur(2px)' }}
            onClick={() => setSelectedLog(null)}
          />
          <div
            className="audit-detail-drawer"
            style={{
              position: 'fixed', top: 0, right: 0, width: 'min(500px, 100vw)', height: '100%', background: 'var(--bg-card)',
              borderLeft: '1px solid var(--border)', zIndex: 1001, boxShadow: 'var(--shadow-lg)',
              display: 'flex', flexDirection: 'column', animation: 'slideIn 0.3s ease-out'
            }}
          >
            <div
              className="audit-detail-header"
              style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}
            >
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>Log Detail</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>
                  {selectedLog.event_type?.replace(/_/g, ' ').toUpperCase()}
                </div>
              </div>
              <button className="btn btn-secondary btn-xs" onClick={() => setSelectedLog(null)}>Close</button>
            </div>

            <div className="audit-detail-body" style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
              <div style={{ marginBottom: 24, padding: 16, background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--green)', fontWeight: 600, fontSize: 12, marginBottom: 16, flexWrap: 'wrap' }}>
                  <RefreshCw size={14} /> SOVEREIGN LEDGER INTEGRITY CHAIN
                </div>
                <div style={{ position: 'relative', paddingLeft: 12, borderLeft: '2px solid var(--border)' }}>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>PREVIOUS BLOCK HASH</div>
                    <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(127, 140, 170, 0.08)', padding: 8, borderRadius: 4, color: 'var(--text-dim)', wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                      {selectedLog.prev_hash || "00000...00"}
                    </div>
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 10, color: 'var(--green)', marginBottom: 4 }}>CURRENT CHAIN HASH</div>
                    <div style={{
                      fontSize: 10, fontFamily: 'var(--font-mono)', background: 'var(--green-light)',
                      padding: 8, borderRadius: 4, color: 'var(--green)', border: '1px solid rgba(16, 185, 129, 0.24)', wordBreak: 'break-word', overflowWrap: 'anywhere'
                    }}>
                      {selectedLog.chain_hash}
                    </div>
                  </div>
                </div>
              </div>
              <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, padding: 16, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                {JSON.stringify(selectedLog.details, null, 2)}
              </div>
            </div>
          </div>
          <style dangerouslySetInnerHTML={{ __html: `
            @keyframes slideIn { from { transform: translateX(100%); } to { transform: translateX(0); } }
            .hover-row:hover td { background-color: var(--accent-light) !important; }
            .audit-detail-drawer { max-width: 100vw; }
            @media (max-width: 768px) {
              .audit-detail-drawer {
                width: 100vw !important;
                border-left: none !important;
              }
              .audit-detail-header {
                padding: 16px !important;
                align-items: flex-start !important;
                flex-wrap: wrap;
              }
              .audit-detail-body {
                padding: 16px !important;
              }
            }
          `}} />
        </>
      )}
    </div>
  )
}
