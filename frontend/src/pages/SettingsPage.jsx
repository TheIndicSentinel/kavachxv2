import { useState, useEffect } from 'react'
import Card from '../components/shared/Card'
import { useAuth } from '../context/AuthContext'
import { settingsAPI } from '../utils/api'
import { Users, Shield, Server, Key, Save, Check } from 'lucide-react'

// Local mock data since we removed it from AuthContext to clean it up
const DEMO_USERS = [
  { id:'adm', name:'Admin User', email:'admin@kavachx.ai', role:'super_admin', roleLabel:'Super Admin', avatar:'AU', org:'KavachX HQ' },
  { id:'cmp', name:'Priya Sharma', email:'compliance@kavachx.ai', role:'compliance_officer', roleLabel:'Compliance', avatar:'PS', org:'Risk & Legal' },
  { id:'eng', name:'Arjun Dev', email:'engineer@kavachx.ai', role:'ml_engineer', roleLabel:'ML Engineer', avatar:'AD', org:'AI Lab' },
]

const ROLE_CAPS = {
  super_admin: { canManageRoles: true, canReadEvents: true, canConfigureSystem: true },
  compliance_officer: { canManageRoles: false, canReadEvents: true, canWritePolicies: true, canReadModels: true },
  ml_engineer: { canReadEvents: true, canRegisterModels: true, canRunSimulations: true },
}

const ROLE_COLORS = { admin:'#3b82f6', executive:'#8b5cf6', engineer:'#06b6d4', compliance:'#10b981', auditor:'#f59e0b' }

export default function SettingsPage() {
  const { user } = useAuth()
  const [saved, setSaved] = useState(false)
  const [thresholds, setThresholds] = useState({
    risk_high: 0.75, risk_medium: 0.45, fairness_disparity: 0.20, confidence_low: 0.60
  })

  useEffect(() => {
    settingsAPI.getThresholds().then(r => {
      if(r.data) {
        setThresholds(r.data)
      }
    }).catch(e => console.error(e))
  }, [])

  const handleSave = async () => {
    try {
      await settingsAPI.updateThresholds(thresholds)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch(e) {
      console.error(e)
    }
  }

  return (
    <div>
      <div className="grid-2" style={{ gap:16, marginBottom:16 }}>
        {/* Governance Thresholds */}
        <Card title="Governance Thresholds" action={<Shield size={14} color="var(--accent)"/>}>
          <div style={{ display:'flex',flexDirection:'column',gap:16 }}>
            <div className="alert alert-info" style={{ marginBottom:0, padding:'10px 14px' }}>
              <div style={{ fontSize:11, lineHeight:1.5 }}>
                <strong>How this affects the app:</strong> These values dictate the High/Medium/Low risk labels on the Dashboard and determine when a policy is flagged in the Audit logs.
              </div>
            </div>

            {[
              { key:'risk_high', label:'High Risk Threshold' },
              { key:'risk_medium', label:'Medium Risk Threshold' },
              { key:'fairness_disparity', label:'Fairness Disparity Max' },
              { key:'confidence_low', label:'Low Confidence Threshold' },
            ].map(f => (
              <div key={f.key}>
                <div style={{ display:'flex',justifyContent:'space-between',marginBottom:4 }}>
                  <div style={{ fontSize:13,fontWeight:500,color:'var(--text)' }}>{f.label}</div>
                  <span style={{ fontFamily:'var(--font-mono)',fontSize:13,fontWeight:600,color:'var(--accent)' }}>{(thresholds[f.key] * 100).toFixed(0)}%</span>
                </div>
                <input type="range" min="0" max="1" step="0.01" value={thresholds[f.key]}
                  onChange={e=>setThresholds(t=>({...t,[f.key]:parseFloat(e.target.value)}))}
                  style={{ width:'100%',accentColor:'var(--accent)', cursor:'pointer' }}
                />
              </div>
            ))}

            <button className="btn btn-primary btn-sm" onClick={handleSave} style={{ alignSelf:'flex-end' }}>
              {saved?<><Check size={13}/>Applied Globally</>:<><Save size={13}/>Save and Update Engine</>}
            </button>
          </div>
        </Card>

        {/* System Info */}
        <Card title="Kavach Node Details" action={<Server size={14} color="var(--accent)"/>}>
          <div style={{ display:'flex',flexDirection:'column',gap:10 }}>
            {[
              ['Deployment ID', 'KAVACH-NODE-IN-001'],
              ['Engine Version', '2.0.0-PROD'],
              ['Kavach Endpoint', 'kavachx-api-node'],
              ['Jurisdiction', 'India (MeitY)'],
              ['Compliance', 'DPDPA, EU AI Act'],
              ['Uptime', '99.9%'],
            ].map(([k,v]) => (
              <div key={k} style={{ display:'flex',justifyContent:'space-between',alignItems:'center',padding:'6px 0',borderBottom:'1px solid var(--border)' }}>
                <span style={{ fontSize:12,color:'var(--text-dim)' }}>{k}</span>
                <span style={{ fontSize:11,fontFamily:'var(--font-mono)',color:'var(--text)', fontWeight:600 }}>{v}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* User Management */}
      <Card title="RBAC & Account Management" action={<Users size={14} color="var(--accent)"/>}>
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Org</th><th>Live Capabilities</th></tr></thead>
            <tbody>
              {DEMO_USERS.map(u => {
                const caps = ROLE_CAPS[u.role]||{}
                const capList = Object.entries(caps).filter(([,v])=>v).map(([k])=>k.replace('can','').replace(/([A-Z])/g,' $1').trim())
                return (
                  <tr key={u.id}>
                    <td>
                      <div style={{ display:'flex',alignItems:'center',gap:8 }}>
                        <div style={{ width:26,height:26,borderRadius:6,background:ROLE_COLORS[u.role]+'20',border:'1px solid '+ROLE_COLORS[u.role]+'40',display:'flex',alignItems:'center',justifyContent:'center',fontSize:9,fontWeight:700,color:ROLE_COLORS[u.role],flexShrink:0 }}>{u.avatar}</div>
                        <span style={{ fontSize:13,fontWeight:500,color:'var(--text)' }}>{u.name}</span>
                      </div>
                    </td>
                    <td style={{ fontFamily:'var(--font-mono)',fontSize:11 }}>{u.email}</td>
                    <td><span style={{ fontSize:10,padding:'2px 8px',borderRadius:99,background:ROLE_COLORS[u.role]+'20',color:ROLE_COLORS[u.role],border:'1px solid '+ROLE_COLORS[u.role]+'30', fontWeight:600 }}>{u.roleLabel}</span></td>
                    <td style={{ fontSize:12 }}>{u.org}</td>
                    <td style={{ fontSize:10,color:'var(--text-muted)',maxWidth:200 }}>{capList.slice(0,3).join(', ')}{capList.length>3?`+${capList.length-3} more`:''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* API Keys */}
      <Card title="Kavach API Access & Connectivity">
        <div className="alert alert-info" style={{ marginBottom:16 }}>
          <span style={{ fontSize:12 }}>These keys allow your external LLMs/ML models to connect to the KavachX Governance Layer.</span>
        </div>
        <div style={{ display:'flex',flexDirection:'column',gap:8 }}>
          {[
            { name:'Production Kavach Core Key', key:'kavachx-demo-key', status:'active', role:'ML Engineer (System)' },
          ].map(k => (
            <div key={k.name} style={{ display:'flex',alignItems:'center',justifyContent:'space-between',padding:'12px 16px',background:'var(--bg-elevated)',border:'1px solid var(--border)',borderRadius:8 }}>
              <div>
                <div style={{ fontSize:13,fontWeight:600,color:'var(--text)' }}>{k.name}</div>
                <div style={{ display:'flex',alignItems:'center',gap:10,marginTop:4 }}>
                  <code style={{ fontFamily:'var(--font-mono)',fontSize:11.5,color:'var(--accent)',background:'var(--accent-light)',padding:'2px 6px',borderRadius:4 }}>{k.key}</code>
                  <button 
                    onClick={() => navigator.clipboard.writeText(k.key)}
                    style={{ fontSize:11,color:'var(--text-muted)',cursor:'pointer',background:'none',border:'none',textDecoration:'underline' }}>
                    Copy Key
                  </button>
                </div>
              </div>
              <div style={{ textAlign:'right' }}>
                <span className="badge badge-active">{k.status}</span>
                <div style={{ fontSize:10,color:'var(--text-muted)',marginTop:4 }}>Access: {k.role}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
