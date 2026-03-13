import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { useTheme } from '../../context/ThemeContext'
import {
  Shield, LayoutDashboard, Code2, ClipboardList, FileText,
  Database, Activity, PlayCircle, GitBranch, FlaskConical,
  Bell, Sun, Moon, LogOut, X, Settings, Terminal
} from 'lucide-react'

const NAV = [
  {
    section: 'Dashboards',
    items: [
      { path: '/',           label: 'Executive View', icon: LayoutDashboard, perm: 'dashboard:read', roles: ['super_admin','executive'] },
      { path: '/engineer',   label: 'ML Engineer',    icon: Code2,           perm: 'dashboard:read', roles: ['super_admin','ml_engineer'] },
      { path: '/compliance', label: 'Compliance',     icon: ClipboardList,   perm: 'dashboard:read', roles: ['super_admin','compliance_officer','auditor'] },
    ],
  },
  {
    section: 'Governance',
    items: [
      { path: '/policies', label: 'Policies',       icon: FileText,    perm: 'policies:read', roles: ['super_admin','compliance_officer','auditor'] },
      { path: '/models',   label: 'Model Registry', icon: Database,    perm: 'models:read',   roles: ['super_admin','ml_engineer','compliance_officer','auditor'] },
      { path: '/audit',    label: 'Audit Logs',     icon: Activity,    perm: 'audit:read',    roles: ['super_admin','compliance_officer','auditor'] },
      { path: '/simulate', label: 'Batch Simulate', icon: PlayCircle,  perm: 'simulate:run',  roles: ['super_admin','ml_engineer'] },
      { path: '/playground', label: 'Kavach Playground', icon: Terminal,   perm: 'dashboard:read' },
    ],
  },
  {
    section: 'Advanced',
    items: [
      { path: '/lineage',     label: 'Data Lineage',      icon: GitBranch,    perm: 'models:read', roles: ['super_admin','ml_engineer'] },
      { path: '/adversarial', label: 'Adversarial Tests', icon: FlaskConical, perm: 'models:read', roles: ['super_admin','ml_engineer'] },
      { path: '/alerts',      label: 'Alerts',            icon: Bell,         perm: 'alerts:read', roles: ['super_admin','ml_engineer','compliance_officer'] },
      { path: '/settings',    label: 'Settings',          icon: Settings,     perm: 'dashboard:read', roles: ['super_admin', 'ml_engineer', 'compliance_officer'] },
    ],
  },
]

export default function Sidebar({ mobileOpen, onClose }) {
  const { user, logout, hasPermission } = useAuth()
  const { dark, toggle } = useTheme()
  const loc = useLocation()
  const nav = useNavigate()
  const role = user?.role || ''

  const canSee = (item) =>
    (hasPermission(item.perm) || hasPermission('*')) &&
    (!item.roles?.length || item.roles.includes(role))

  const go = (path) => { nav(path); onClose?.() }

  return (
    <>
      <div className={`sidebar-overlay ${mobileOpen ? 'open' : ''}`} onClick={onClose} />

      <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
        {/* Brand */}
        <div className="sidebar-logo" style={{ alignItems: 'flex-start', paddingTop: 16 }}>
          <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='28' height='28' fill='url(#grad)' stroke='none' style={{ flexShrink: 0, marginTop: 1 }}>
            <defs>
              <linearGradient id='grad' x1='0%' y1='0%' x2='100%' y2='100%'>
                <stop offset='0%' stopColor='#a5b4fc' />
                <stop offset='50%' stopColor='#4f46e5' />
                <stop offset='100%' stopColor='#1e1b4b' />
              </linearGradient>
            </defs>
            <path d='M12 1 L1 5 V12 C1 18 6 22.5 12 24 C18 22.5 23 18 23 12 V5 Z'/>
          </svg>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="sidebar-logo-text">KavachX</div>
            <div className="sidebar-logo-sub">AI Governance v2</div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close sidebar"
            style={{
              display: mobileOpen ? 'flex' : 'none',
              alignItems: 'center', justifyContent: 'center',
              width: 26, height: 26,
              color: 'var(--text-muted)', background: 'none',
              border: 'none', cursor: 'pointer',
              borderRadius: 5, flexShrink: 0,
            }}
          >
            <X size={14} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {NAV.map((section) => {
            const visible = section.items.filter(canSee)
            if (!visible.length) return null
            return (
              <div key={section.section}>
                <div className="sidebar-section-label">{section.section}</div>
                {visible.map((item) => (
                  <div
                    key={item.path}
                    className={`sidebar-item ${loc.pathname === item.path ? 'active' : ''}`}
                    onClick={() => go(item.path)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => e.key === 'Enter' && go(item.path)}
                  >
                    <item.icon size={14} className="sidebar-item-icon" />
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer">
          <div className="sidebar-item" onClick={toggle} role="button" tabIndex={0}>
            {dark ? <Sun size={14} className="sidebar-item-icon" /> : <Moon size={14} className="sidebar-item-icon" />}
            <span>{dark ? 'Light Mode' : 'Dark Mode'}</span>
          </div>

          {user && (
            <div className="sidebar-user">
              <div className="sidebar-avatar">
                {(user.avatar || user.name?.[0] || '?').toUpperCase()}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="sidebar-user-name">{user.name}</div>
                <div className="sidebar-user-role">{user.role_label || user.role}</div>
              </div>
            </div>
          )}

          <div
            className="sidebar-item"
            onClick={() => { logout(); onClose?.() }}
            role="button" tabIndex={0}
          >
            <LogOut size={14} className="sidebar-item-icon" />
            <span>Sign out</span>
          </div>
        </div>
      </aside>
    </>
  )
}
