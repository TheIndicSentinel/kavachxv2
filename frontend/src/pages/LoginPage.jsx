import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import {
  Sun, Moon, Eye, EyeOff, ArrowRight, ShieldCheck,
  KeyRound, UserPlus, LogIn, Check, X as XIcon, Loader2
} from 'lucide-react'

// Password strength checker
function checkPassword(pw) {
  const rules = [
    { id: 'len',   label: 'At least 12 characters',      ok: pw.length >= 12 },
    { id: 'upper', label: 'Uppercase letter (A–Z)',       ok: /[A-Z]/.test(pw) },
    { id: 'lower', label: 'Lowercase letter (a–z)',       ok: /[a-z]/.test(pw) },
    { id: 'digit', label: 'Number (0–9)',                  ok: /\d/.test(pw) },
    { id: 'spec',  label: 'Special character (!@#$…)',    ok: /[!@#$%^&*()\-_=+\[\]{}|;:'",.<>?/`~\\]/.test(pw) },
  ]
  const score = rules.filter(r => r.ok).length
  return { rules, score, valid: score === 5 }
}

function PasswordStrengthBar({ password }) {
  if (!password) return null
  const { rules, score } = checkPassword(password)
  const colors = ['var(--red)', 'var(--red)', 'var(--amber)', 'var(--amber)', 'var(--green)', 'var(--green)']
  const color = colors[score] || 'var(--border)'
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', gap: 3, marginBottom: 6 }}>
        {[1,2,3,4,5].map(i => (
          <div key={i} style={{
            flex: 1, height: 3, borderRadius: 2,
            background: i <= score ? color : 'var(--border)',
            transition: 'background 0.2s',
          }} />
        ))}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {rules.map(r => (
          <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10.5 }}>
            {r.ok
              ? <Check size={10} style={{ color: 'var(--green)', flexShrink: 0 }} />
              : <XIcon size={10} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            }
            <span style={{ color: r.ok ? 'var(--text-dim)' : 'var(--text-muted)' }}>{r.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PasswordField({ value, onChange, placeholder, label, showMeter = false }) {
  const [show, setShow] = useState(false)
  return (
    <div className="form-group">
      {label && <label className="form-label">{label}</label>}
      <div style={{ position: 'relative' }}>
        <input
          className="form-input"
          type={show ? 'text' : 'password'}
          placeholder={placeholder || '••••••••••••'}
          value={value}
          onChange={e => onChange(e.target.value)}
          style={{ paddingRight: 38 }}
          required
        />
        <button type="button" onClick={() => setShow(s => !s)}
          style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer' }}>
          {show ? <EyeOff size={15} /> : <Eye size={15} />}
        </button>
      </div>
      {showMeter && <PasswordStrengthBar password={value} />}
    </div>
  )
}

export default function LoginPage() {
  const { login, bootstrap, setupStatus } = useAuth()
  const { dark, toggle } = useTheme()
  const nav      = useNavigate()
  const location = useLocation()

  const [tab, setTab]     = useState('login')    // 'login' | 'setup'
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Login fields
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')

  // Setup (bootstrap) fields
  const [setupName,    setSetupName]    = useState('')
  const [setupEmail,   setSetupEmail]   = useState('')
  const [setupPw,      setSetupPw]      = useState('')
  const [setupPw2,     setSetupPw2]     = useState('')
  const [setupToken,   setSetupToken]   = useState('')
  const [setupSuccess, setSetupSuccess] = useState(false)
  const [showToken,    setShowToken]    = useState(false)

  // Auto-switch to setup tab if first run
  useEffect(() => {
    if (setupStatus?.setup_required) setTab('setup')
  }, [setupStatus])

  // Pre-fill bootstrap token from ?setup_token= query param (console quick-setup URL)
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const token  = params.get('setup_token')
    if (token) {
      setSetupToken(token)
      setTab('setup')
      // Clean the URL without reloading the page (avoid token leaking in history)
      const url = new URL(window.location.href)
      url.searchParams.delete('setup_token')
      window.history.replaceState({}, '', url.pathname + (url.search || ''))
    }
  }, [location.search])

  const handleLogin = async (e) => {
    e?.preventDefault()
    if (!email || !password) return setError('Email and password are required')
    setLoading(true); setError('')
    try {
      await login(email, password)
      nav('/')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Login failed')
    } finally { setLoading(false) }
  }

  const handleBootstrap = async (e) => {
    e?.preventDefault()
    if (!setupName.trim() || !setupEmail.trim() || !setupPw || !setupToken.trim()) {
      return setError('All fields are required')
    }
    if (setupPw !== setupPw2) return setError('Passwords do not match')
    const { valid } = checkPassword(setupPw)
    if (!valid) return setError('Password does not meet the requirements below')
    setLoading(true); setError('')
    try {
      await bootstrap({ name: setupName.trim(), email: setupEmail.trim(), password: setupPw, bootstrap_token: setupToken.trim() })
      setSetupSuccess(true)
      setTimeout(() => nav('/'), 1500)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (typeof detail === 'object' && detail?.password_errors) {
        setError(detail.password_errors.join(' · '))
      } else {
        setError(typeof detail === 'string' ? detail : err.message || 'Setup failed')
      }
    } finally { setLoading(false) }
  }

  const setupRequired = setupStatus?.setup_required === true
  const loadingStatus = setupStatus === null

  return (
    <div className="login-page">
      <button className="topbar-btn" onClick={toggle}
        style={{ position: 'fixed', top: 16, right: 16, zIndex: 10 }}>
        {dark ? <Sun size={15} /> : <Moon size={15} />}
      </button>

      <div className="login-card fade-up">
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 11, marginBottom: 24, paddingTop: 4 }}>
          <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='38' height='38' fill='url(#kx-grad)' stroke='none' style={{ flexShrink: 0, marginTop: 2 }}>
            <defs>
              <linearGradient id='kx-grad' x1='0%' y1='0%' x2='100%' y2='100%'>
                <stop offset='0%' stopColor='#a5b4fc' />
                <stop offset='50%' stopColor='#4f46e5' />
                <stop offset='100%' stopColor='#1e1b4b' />
              </linearGradient>
            </defs>
            <path d='M12 1 L1 5 V12 C1 18 6 22.5 12 24 C18 22.5 23 18 23 12 V5 Z'/>
          </svg>
          <div>
            <div style={{ fontSize: 21, fontWeight: 800, letterSpacing: '-.04em' }}>KavachX</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '.08em' }}>AI GOVERNANCE PLATFORM</div>
          </div>
        </div>

        {loadingStatus ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
            <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} />
          </div>
        ) : (
          <>
            {/* Tab switcher — only show setup tab if first run */}
            {setupRequired && (
              <div style={{
                display: 'flex', gap: 0, marginBottom: 20,
                background: 'var(--bg-elevated)', borderRadius: 9, padding: 3,
                border: '1px solid var(--border)',
              }}>
                {[
                  { id: 'login', icon: LogIn,    label: 'Sign In' },
                  { id: 'setup', icon: KeyRound,  label: 'First Run Setup' },
                ].map(t => (
                  <button key={t.id} onClick={() => { setTab(t.id); setError('') }}
                    style={{
                      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                      padding: '7px 12px', borderRadius: 7, border: 'none', cursor: 'pointer',
                      fontSize: 12.5, fontWeight: 600, fontFamily: 'var(--font)',
                      transition: 'all .13s',
                      background: tab === t.id ? 'var(--bg-card)' : 'transparent',
                      color: tab === t.id ? 'var(--text)' : 'var(--text-muted)',
                      boxShadow: tab === t.id ? 'var(--shadow-xs)' : 'none',
                    }}>
                    <t.icon size={13} />{t.label}
                  </button>
                ))}
              </div>
            )}

            {error && (
              <div className="alert alert-error" style={{ marginBottom: 14 }}>
                <span style={{ fontSize: 12.5 }}>{error}</span>
              </div>
            )}

            {/* ── SIGN IN ── */}
            {tab === 'login' && (
              <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input className="form-input" type="email" placeholder="you@organization.com"
                    value={email} onChange={e => setEmail(e.target.value)} required autoComplete="email" />
                </div>
                <PasswordField value={password} onChange={setPassword} label="Password" placeholder="••••••••••••" />
                <button type="submit" className="btn btn-primary w-full" disabled={loading}
                  style={{ justifyContent: 'center', marginTop: 4, padding: '10px 16px' }}>
                  {loading
                    ? <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Signing in…</>
                    : <><span>Sign In</span><ArrowRight size={14} /></>
                  }
                </button>
              </form>
            )}

            {/* ── FIRST RUN SETUP ── */}
            {tab === 'setup' && (
              <>
                {setupSuccess ? (
                  <div style={{ textAlign: 'center', padding: '24px 0' }}>
                    <ShieldCheck size={40} style={{ color: 'var(--green)', margin: '0 auto 12px' }} />
                    <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text)', marginBottom: 6 }}>
                      Super Admin created!
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Redirecting to dashboard…</div>
                  </div>
                ) : (
                  <>
                    <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--accent-10)', border: '1px solid var(--accent-30)', marginBottom: 16 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', marginBottom: 3 }}>First Run — Create Super Admin</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.65 }}>
                        Enter the bootstrap token printed in your server logs to create the initial Super Admin account.
                        This can only be done once.
                      </div>
                    </div>

                    <form onSubmit={handleBootstrap} style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
                      <div className="form-group">
                        <label className="form-label">Full Name</label>
                        <input className="form-input" type="text" placeholder="Your full name"
                          value={setupName} onChange={e => setSetupName(e.target.value)} required />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Email</label>
                        <input className="form-input" type="email" placeholder="admin@organization.com"
                          value={setupEmail} onChange={e => setSetupEmail(e.target.value)} required autoComplete="email" />
                      </div>

                      <div className="form-group">
                        <label className="form-label">Bootstrap Token</label>
                        <div style={{ position: 'relative' }}>
                          <input
                            className="form-input"
                            type={showToken ? 'text' : 'password'}
                            placeholder="Paste token from server logs"
                            value={setupToken}
                            onChange={e => setSetupToken(e.target.value)}
                            style={{ paddingRight: 38, fontFamily: 'var(--font-mono)', fontSize: 12 }}
                            required
                          />
                          <button type="button" onClick={() => setShowToken(s => !s)}
                            style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer' }}>
                            {showToken ? <EyeOff size={15} /> : <Eye size={15} />}
                          </button>
                        </div>
                      </div>

                      <PasswordField value={setupPw} onChange={setSetupPw} label="Password" showMeter />

                      <div className="form-group">
                        <label className="form-label">Confirm Password</label>
                        <div style={{ position: 'relative' }}>
                          <input className="form-input" type="password"
                            placeholder="Repeat password"
                            value={setupPw2} onChange={e => setSetupPw2(e.target.value)} required />
                        </div>
                        {setupPw2 && setupPw !== setupPw2 && (
                          <span style={{ fontSize: 11, color: 'var(--red)' }}>Passwords do not match</span>
                        )}
                      </div>

                      <button type="submit" className="btn btn-primary w-full"
                        disabled={loading || (setupPw2 && setupPw !== setupPw2) || !checkPassword(setupPw).valid}
                        style={{ justifyContent: 'center', marginTop: 4, padding: '10px 16px' }}>
                        {loading
                          ? <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Creating account…</>
                          : <><ShieldCheck size={14} /><span>Create Super Admin</span></>
                        }
                      </button>
                    </form>
                  </>
                )}
              </>
            )}
          </>
        )}

        <div style={{ marginTop: 20, fontSize: 10.5, color: 'var(--text-muted)', textAlign: 'center', lineHeight: 1.6 }}>
          KavachX · India-first AI Governance &nbsp;·&nbsp;
          <span style={{ color: 'var(--accent-2)' }}>EU AI Act</span> &amp;{' '}
          <span style={{ color: 'var(--accent-2)' }}>DPDP 2023</span> ready
        </div>

        <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
      </div>
    </div>
  )
}
