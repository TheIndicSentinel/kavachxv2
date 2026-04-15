import { useEffect, useState, useCallback } from 'react'
import { policiesAPI } from '../utils/api'
import { useAuth } from '../context/AuthContext'
import {
  Plus, Trash2, ToggleLeft, ToggleRight, X, Shield, Info,
  Upload, ChevronLeft, ChevronRight, Code, FileJson, BookOpen,
  Lightbulb, Globe, Lock, Eye, Users, AlertTriangle
} from 'lucide-react'

/* ─────────────────────────────────────────────────────────────
   CONSTANTS & HELPERS
───────────────────────────────────────────────────────────── */
const isBaseline = (p) => String(p.id).startsWith('builtin')
const isPrinciple = (p) => String(p.id).startsWith('principle')

const TYPE_MAP = {
  fairness: 'badge-info',
  safety: 'badge-block',
  compliance: 'badge-alert',
  performance: 'badge-pass',
  llm_safety: 'badge-review',
  principle: 'badge-principle',
}
const SEV_MAP = {
  critical: 'badge-critical',
  high: 'badge-alert',
  medium: 'badge-warning',
  low: 'badge-muted',
}
const POLICY_TYPES = ['fairness', 'safety', 'compliance', 'performance', 'llm_safety']
const SEVERITIES = ['low', 'medium', 'high', 'critical']
const JURISDICTIONS = ['GLOBAL', 'IN', 'EU', 'US', 'APAC']
const PAGE_SIZE = 12

/* ─────────────────────────────────────────────────────────────
   BASELINE POLICIES — India-first, then global
   Derived from: DPDP Act 2023, RBI AI Guidelines, SEBI Tech Risk,
   IRDAI Sandbox Regulations, MeitY AI Policy Advisory, IT Act 2000
   (as amended), NHA/ABDM data standards, and global baselines.
───────────────────────────────────────────────────────────── */
const BASELINE_POLICIES = [
  /* ── Indian Regulatory Baseline ── */
  {
    id: 'builtin-dpdp-001',
    name: 'DPDP Act — Consent Verification Gate',
    description: 'Block any AI inference that processes personal data without verified, purpose-specific consent as mandated by the Digital Personal Data Protection Act 2023. Consent metadata must be present and timestamp-valid before inference proceeds.',
    policy_type: 'compliance',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'DPDP Act 2023 § 6',
  },
  {
    id: 'builtin-dpdp-002',
    name: 'DPDP Act — Data Minimisation Enforcement',
    description: 'Flag models that consume feature sets beyond the declared purpose of data collection. Enforces the data minimisation principle under DPDP Act 2023: no more data than is necessary for the stated processing purpose.',
    policy_type: 'compliance',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'DPDP Act 2023 § 6(b)',
  },
  {
    id: 'builtin-dpdp-003',
    name: 'DPDP Act — Data Principal Rights Audit',
    description: 'Ensure AI systems can produce machine-readable explanations of decisions affecting data principals, and that correction/erasure requests can be operationalised. Required for automated decision-making under DPDP Act 2023.',
    policy_type: 'compliance',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'DPDP Act 2023 § 12–14',
  },
  {
    id: 'builtin-rbi-001',
    name: 'RBI Fairness Check — Credit Demographic Parity',
    description: 'Demographic parity for credit decisions across RBI-defined protected groups. Maximum permissible disparity is 10% across gender, religion, caste (proxied), and regional origin. Enforces RBI AI Governance Guidelines for financial services AI.',
    policy_type: 'fairness',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'RBI AI Guidelines 2024',
  },
  {
    id: 'builtin-rbi-002',
    name: 'RBI — Model Risk Management Disclosure',
    description: 'All AI models used in credit underwriting, fraud detection, or customer risk scoring must have documented model risk assessments filed with the system of record. Enforces RBI Master Direction on IT Risk and Cyber Security.',
    policy_type: 'compliance',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'RBI Master Direction 2021',
  },
  {
    id: 'builtin-rbi-003',
    name: 'RBI — Debt-to-Income Ratio Compliance',
    description: 'Flag AI-driven loan decisions where predicted debt-to-income ratio exceeds 40% without documented human override. Prevents automated approval of structurally unaffordable credit products.',
    policy_type: 'compliance',
    severity: 'medium',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'RBI Circular DOR.CRE.REC.5/21.01.001/2022-23',
  },
  {
    id: 'builtin-sebi-001',
    name: 'SEBI — Algorithmic Decision Audit Trail',
    description: 'All AI-driven investment recommendations, portfolio adjustments, or risk assessments must generate tamper-evident audit logs retained for 5 years. Enforces SEBI Circular on Algorithmic Trading and AI-based advisory systems.',
    policy_type: 'compliance',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'SEBI Circular SEBI/HO/MIRSD/2024',
  },
  {
    id: 'builtin-irdai-001',
    name: 'IRDAI — Insurance AI Explainability Mandate',
    description: 'AI models used in insurance underwriting or claim settlement must produce explainable outputs in a form communicable to policyholders in their preferred language. Enforces IRDAI Guidelines on Use of AI in Insurance 2023.',
    policy_type: 'fairness',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'IRDAI Guidelines 2023',
  },
  {
    id: 'builtin-meity-001',
    name: 'MeitY — High-Risk AI Domain Classification',
    description: 'Models deployed in healthcare, judicial, welfare benefits, and public safety domains are classified as high-risk. Requires ≥70% confidence threshold, mandatory human review gate, and quarterly independent bias audit. Aligned with MeitY AI Advisory and India\'s draft National AI Policy.',
    policy_type: 'compliance',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'MeitY AI Advisory 2023',
  },
  {
    id: 'builtin-it-001',
    name: 'IT Act 2000 — Intermediary Liability for AI Outputs',
    description: 'AI-generated content published via intermediary platforms must be screened for content that violates IT Act § 79 safe harbour conditions — including defamatory, obscene, or incitement-to-violence content. Loss of safe harbour protection triggers BLOCK action.',
    policy_type: 'llm_safety',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'IT Act 2000 § 79; IT Rules 2021',
  },
  {
    id: 'builtin-nha-001',
    name: 'NHA/ABDM — Health Data Sovereignty Check',
    description: 'AI models processing health records linked to Ayushman Bharat Digital Mission (ABDM) must verify that data access is consent-gated via the Health Data Management Policy. Cross-border health data transfer for AI training is blocked pending explicit regulatory clearance.',
    policy_type: 'compliance',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'ABDM Health Data Management Policy 2022',
  },
  /* ── Indian Contextual Fairness Baseline ── */
  {
    id: 'builtin-ctx-001',
    name: 'Caste-Proxy Correlation Guard',
    description: 'Detect and flag AI models whose predictions show statistically significant correlation with caste-proxy indicators: surname clusters, pin code social indices, and institution tier indices. Maximum permissible caste-proxy correlation is 0.08 (Cramér\'s V). Core KavachX India differentiator.',
    policy_type: 'fairness',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'Constitution of India Art. 15; DPDP Act 2023',
  },
  {
    id: 'builtin-ctx-002',
    name: 'Multilingual Performance Equity',
    description: 'AI models processing natural language must achieve accuracy parity within ±8% across all Indian scheduled languages present in the deployment population. Models with larger performance gaps are flagged for retraining before production deployment.',
    policy_type: 'fairness',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'MeitY AI Advisory 2023; NEP 2020',
  },
  {
    id: 'builtin-ctx-003',
    name: 'Informal Economy Inclusion Audit',
    description: 'Models targeting Indian consumers must demonstrate adequate prediction quality for individuals without formal employment records, credit history, or documented address trails. Minimum 85% population coverage required. Flags systematic exclusion of informal sector workers.',
    policy_type: 'fairness',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    regulation: 'KavachX Inclusion Standard v1',
  },
  /* ── Global Technical Baseline ── */
  {
    id: 'builtin-sc-001',
    name: 'Low Confidence Gate',
    description: 'Block inferences below 55% confidence in high-stakes domains (credit, healthcare, hiring, benefits). Required across all deployment contexts. Models operating below threshold must escalate to human review.',
    policy_type: 'safety',
    severity: 'high',
    enabled: true,
    jurisdiction: 'GLOBAL',
    regulation: 'KavachX Safety Baseline v1',
  },
  {
    id: 'builtin-llm-001',
    name: 'LLM Toxicity Guard',
    description: 'Block outputs where toxicity score exceeds 60%, or where communal, casteist, or religion-targeting content is detected. Applies to all LLM and generative AI endpoints. India context includes regional language toxicity detection.',
    policy_type: 'llm_safety',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'GLOBAL',
    regulation: 'IT Rules 2021; KavachX LLM Safety Baseline',
  },
  {
    id: 'builtin-perf-001',
    name: 'Model Drift Monitor',
    description: 'Alert when model performance degrades >15% from baseline within 24 hours (PSI > 0.2). Trigger full re-audit if performance drops >25%. Includes seasonal drift detection for India agricultural cycle patterns.',
    policy_type: 'performance',
    severity: 'medium',
    enabled: true,
    jurisdiction: 'GLOBAL',
    regulation: 'KavachX Monitoring Baseline v1',
  },
]

/* ─────────────────────────────────────────────────────────────
   PRINCIPLE POLICIES — Two tiers:
   Tier 1: Derived from Indian laws and regulatory guidance
           where no specific technical standard yet exists.
   Tier 2: Forward-looking governance principles from the
           KavachX vision — not yet in any market tool.
───────────────────────────────────────────────────────────── */
const PRINCIPLE_POLICIES = [
  /* ── Tier 1: India Law & Regulatory Derivation ── */
  {
    id: 'principle-in-001',
    name: 'Aadhar Linkage Audit Principle',
    description: 'AI models that use Aadhaar-linked identifiers for inference must validate that linkage was obtained under the Authentication User Agency (AUA) framework. Any model inferring identity through Aadhaar demographic or biometric data without AUA registration is blocked. Derived from the Supreme Court\'s Puttaswamy judgment and UIDAI regulations.',
    policy_type: 'principle',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'Puttaswamy v. Union of India (2018); UIDAI AUA Agreement',
    rationale: 'No existing technical standard operationalises the Puttaswamy privacy ruling for AI inference contexts. This principle fills that gap.',
  },
  {
    id: 'principle-in-002',
    name: 'Constitutional Non-Discrimination Inference Guard',
    description: 'AI systems making decisions affecting fundamental rights must not produce outcomes that disproportionately disadvantage groups protected under Articles 14, 15, and 16 of the Constitution of India. Disparity exceeding two standard deviations from baseline requires mandatory human review before decision delivery.',
    policy_type: 'principle',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'Constitution of India Art. 14, 15, 16',
    rationale: 'Constitutional protections apply to AI-mediated government decisions but no implementing technical standard exists at the model level. This principle makes the constitutional guarantee machine-enforceable.',
  },
  {
    id: 'principle-in-003',
    name: 'Consumer Protection Act — AI Deceptive Practice Guard',
    description: 'AI-generated recommendations, pricing, or product descriptions must not constitute an unfair trade practice under the Consumer Protection Act 2019. Models that personalise pricing in ways that create systematic disadvantage for identifiable consumer segments are flagged for review.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'Consumer Protection Act 2019 § 2(47); CPA E-Commerce Rules 2020',
    rationale: 'E-commerce AI pricing is a known consumer harm vector. The CPA 2019 prohibits unfair trade practices but provides no technical enforcement mechanism for AI systems.',
  },
  {
    id: 'principle-in-004',
    name: 'RTI-Aligned Algorithmic Transparency Principle',
    description: 'AI systems used by public authorities to make decisions affecting citizens must be capable of producing decision explanations that satisfy the Right to Information Act 2005 disclosure standard: a citizen who files an RTI for the reasoning behind an AI-influenced decision must receive a meaningful, non-technical explanation.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'Right to Information Act 2005 § 4(1)(b)',
    rationale: 'Government AI decisions are technically subject to RTI but no standard for AI-system RTI responses exists. This principle requires models to maintain citizen-interpretable explanation records.',
  },
  {
    id: 'principle-in-005',
    name: 'PM-WANI / Digital Public Infrastructure Inclusion Guard',
    description: 'AI services operating on Digital Public Infrastructure (UPI, ONDC, Aadhaar, ABDM) must maintain service quality parity for users accessing through feature phones, low-bandwidth connections, and assisted-digital channels. Models that degrade in quality for non-smartphone users must be flagged.',
    policy_type: 'principle',
    severity: 'medium',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'TRAI Recommendations on Broadband Equity 2022; PM-WANI Framework',
    rationale: 'India\'s DPI ecosystem was designed for inclusion. AI systems built on DPI must not reintroduce the digital divide through performance inequality across device and connectivity tiers.',
  },
  {
    id: 'principle-in-006',
    name: 'NEP 2020 — EdTech AI Non-Surveillance Principle',
    description: 'AI systems deployed in educational settings must not engage in continuous behavioural surveillance of students without explicit parental consent. Predictive academic profiling that creates permanent risk classifications for minors is blocked. Derived from NEP 2020 data governance principles and NCERT guidelines.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'NEP 2020 § 23–24; NCERT Data Governance Framework',
    rationale: 'EdTech AI surveillance of children is a documented harm. India\'s NEP implies protections that are not yet operationalised in technical policy standards.',
  },
  {
    id: 'principle-in-007',
    name: 'Gig Economy Worker Algorithmic Accountability',
    description: 'AI systems that determine task allocation, earnings, access, or deactivation for gig economy workers must provide workers with an explanation of adverse decisions and a contestation pathway. Worker deactivation by algorithm without human review is blocked. Derived from Code on Social Security 2020 and emerging platform labour jurisprudence.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'IN',
    tier: 1,
    source: 'Code on Social Security 2020; IFAT v. Urban Company (2022)',
    rationale: 'India has over 15 million gig workers governed by algorithmic management. Courts have begun applying labour protections but no technical standard exists for algorithm-driven worker management systems.',
  },
  /* ── Tier 2: Forward-Looking Governance Principles ── */
  {
    id: 'principle-fwd-001',
    name: 'Adversarial Governance Integrity Check',
    description: 'Governance checks are applied via runtime interception on live inference traffic, not batch testing on curated samples. Models must commit to behavioural parameters at registration. Randomised audit probes are injected continuously to detect discrepancies between compliance posture and production behaviour — catching "compliance theatre".',
    policy_type: 'principle',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 4.1, § 8.1',
    rationale: 'No commercial governance tool currently prevents organisations from running compliant models in testing and non-compliant models in production. This principle makes governance gaming architecturally detectable.',
  },
  {
    id: 'principle-fwd-002',
    name: 'Causal Disparity Classification Principle',
    description: 'When fairness metrics detect output disparities, KavachX must classify the disparity as: (a) historical-injustice-correlated — requiring active remediation, (b) genuine-variation — requiring human expert review, or (c) proxy-discrimination — triggering automatic block. Unclassified disparities default to HUMAN_REVIEW. Prevents statistical compliance masking genuine discrimination.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 8.2; Causal Fairness Literature',
    rationale: 'Current fairness tools measure correlation, not causation. This principle enforces causal classification as a governance standard — a frontier capability not present in any current market tool.',
  },
  {
    id: 'principle-fwd-003',
    name: 'Citizen Redress & Contestation Right',
    description: 'Any individual affected by an AI decision in a regulated domain (credit, employment, healthcare, benefits, insurance) has the right to: (1) a plain-language explanation of the decision in their preferred language, (2) a machine-readable audit trail reference, and (3) a contestation pathway with guaranteed human review within 72 hours.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 4.3, § 5; DPDP Act 2023 § 12',
    rationale: 'No existing governance platform includes a citizen-facing redress layer. This principle extends governance from organisational compliance to individual rights — consistent with the DPDP Act\'s intent but beyond its current technical implementation.',
  },
  {
    id: 'principle-fwd-004',
    name: 'Cryptographic Audit Trail Integrity',
    description: 'All governance decisions — enforcement actions, risk scores, policy evaluations — are recorded with cryptographic hash chaining. Retroactive modification of audit records is architecturally impossible. Regulators and third-party auditors are issued direct read-only API access to audit chains without organisational mediation.',
    policy_type: 'principle',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 4.1, § 8.1',
    rationale: 'Current AI audit logs are mutable and organisationally controlled. Cryptographic immutability makes audit trails legally reliable and regulator-accessible — a standard that does not exist in any deployed governance tool.',
  },
  {
    id: 'principle-fwd-005',
    name: 'Inclusion Audit — Data-Sparse Population Coverage',
    description: 'Before production deployment, models must demonstrate prediction quality is statistically comparable for data-sparse and data-rich populations. Systems where ≥15% of the target population falls below minimum prediction confidence thresholds are blocked pending coverage improvement. Prevents systematic exclusion of informal-economy populations.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 3.2, § 4.2',
    rationale: 'Standard bias frameworks test between groups within the training distribution. The inclusion audit tests for groups outside it entirely — a category of governance gap that no existing commercial tool addresses.',
  },
  {
    id: 'principle-fwd-006',
    name: 'Longitudinal Fairness Degradation Detection',
    description: 'Fairness metrics are tracked over time, not only at deployment. A model may pass initial fairness checks but drift into discriminatory behaviour as data distributions shift. Models showing sustained fairness metric degradation over 14-day rolling windows trigger re-audit regardless of current accuracy metrics.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 5 (Enhanced Governance Module Stack)',
    rationale: 'Current governance tools audit at deployment time. Longitudinal fairness tracking — monitoring bias drift the same way accuracy drift is monitored — is absent from all current commercial platforms.',
  },
  {
    id: 'principle-fwd-007',
    name: 'Generative AI Disinformation Scoring',
    description: 'LLM outputs distributed at scale must be scored for disinformation potential before delivery: factual accuracy estimation, misleading framing detection, and coordinated narrative detection. Outputs above disinformation risk threshold require human editorial review before publication. Addresses qualitatively new risk surface of LLMs versus traditional ML.',
    policy_type: 'principle',
    severity: 'high',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 1.4, § 5',
    rationale: 'Hallucination and toxicity checks exist in LLM safety tools but disinformation-at-scale scoring — detecting coordinated misleading narratives — is not implemented in any current governance platform.',
  },
  {
    id: 'principle-fwd-008',
    name: 'Model Accountability Traceability Chain',
    description: 'For every consequential AI decision, the system maintains a full traceability chain: model version → training data lineage → governance evaluation → enforcement action → outcome. This chain constitutes the evidentiary record for regulatory inquiries, legal proceedings, and incident classification. No link in the chain may be deleted.',
    policy_type: 'principle',
    severity: 'critical',
    enabled: true,
    jurisdiction: 'GLOBAL',
    tier: 2,
    source: 'KavachX Vision Framework § 8.3',
    rationale: 'Accountability for AI-caused harm requires an unbroken evidentiary record. Current systems log events but do not maintain legally-grade traceability chains linking decisions to model versions and training data.',
  },
]

const ALL_BASELINE = BASELINE_POLICIES
const ALL_PRINCIPLE = PRINCIPLE_POLICIES

/* ─────────────────────────────────────────────────────────────
   SAMPLE JSON TEMPLATE
───────────────────────────────────────────────────────────── */
const SAMPLE_JSON = {
  name: "My Custom Policy",
  description: "Describe what this policy enforces and why.",
  policy_type: "fairness",
  severity: "medium",
  jurisdiction: "IN",
  rules: [
    { field: "confidence", operator: "lt", value: 0.6, action: "ALERT" },
    { field: "risk_score", operator: "gt", value: 0.8, action: "BLOCK" }
  ]
}

/* ─────────────────────────────────────────────────────────────
   PAGINATION
───────────────────────────────────────────────────────────── */
function Pagination({ page, total, pageSize, onChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  if (totalPages <= 1) return null
  const window5 = Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
    return Math.max(1, Math.min(totalPages - 4, page - 2)) + i
  })
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 0 0', borderTop: '1px solid var(--border)', marginTop: 8 }}>
      <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
        Showing {Math.min((page - 1) * pageSize + 1, total)}–{Math.min(page * pageSize, total)} of {total}
      </span>
      <div style={{ display: 'flex', gap: 4 }}>
        <button className="btn btn-secondary btn-xs" onClick={() => onChange(page - 1)} disabled={page === 1}><ChevronLeft size={13} /></button>
        {window5.map(p => (
          <button key={p} className={`btn btn-xs ${p === page ? 'btn-primary' : 'btn-secondary'}`} onClick={() => onChange(p)}>{p}</button>
        ))}
        <button className="btn btn-secondary btn-xs" onClick={() => onChange(page + 1)} disabled={page === totalPages}><ChevronRight size={13} /></button>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   POLICY CARD
───────────────────────────────────────────────────────────── */
function PolicyCard({ policy, canToggle, canDelete, onToggle, onDelete, toggling, deleting }) {
  const [expanded, setExpanded] = useState(false)
  const baseline = isBaseline(policy)
  const principle = isPrinciple(policy)
  const isTogglingThis = toggling === policy.id
  const isDeletingThis = deleting === policy.id

  const accentColor = principle
    ? (policy.tier === 2 ? 'var(--purple)' : 'var(--cyan)')
    : 'var(--accent)'

  const isCritical = policy.severity === 'critical' && policy.enabled

  return (
    <div
      className={isCritical ? 'item-critical' : ''}
      style={{
        padding: '13px 16px',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border)',
        background: 'var(--bg-elevated)',
        opacity: policy.enabled ? 1 : 0.58,
        borderLeft: `3px solid ${policy.enabled ? (isCritical ? 'var(--red)' : accentColor) : 'var(--border)'}`,
        transition: 'all .15s',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
        {/* Toggle */}
        <button
          onClick={() => onToggle(policy)}
          disabled={!canToggle || baseline || principle || isTogglingThis}
          title={
            baseline || principle ? 'Baseline and Principle policies cannot be disabled'
              : !canToggle ? 'Super Admin required'
                : policy.enabled ? 'Disable policy' : 'Enable policy'
          }
          style={{
            background: 'none', border: 'none',
            cursor: canToggle && !baseline && !principle ? 'pointer' : 'default',
            color: policy.enabled ? accentColor : 'var(--text-muted)',
            padding: 0, flexShrink: 0, marginTop: 1,
            opacity: (!canToggle || baseline || principle) ? 0.35 : 1,
          }}
        >
          {isTogglingThis
            ? <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
            : policy.enabled ? <ToggleRight size={22} /> : <ToggleLeft size={22} />
          }
        </button>

        {/* Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap', marginBottom: 4 }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>{policy.name}</span>

            {baseline && <span className="badge badge-baseline"><Shield size={9} /> Baseline</span>}
            {principle && policy.tier === 1 && (
              <span className="badge badge-principle-law"><BookOpen size={9} /> Principle · Law-Derived</span>
            )}
            {principle && policy.tier === 2 && (
              <span className="badge badge-principle-fwd"><Lightbulb size={9} /> Principle · Forward-Looking</span>
            )}

            <span className={`badge ${TYPE_MAP[policy.policy_type] || 'badge-muted'}`}>
              {policy.policy_type?.replace('_', ' ')}
            </span>
            <span className={`badge ${SEV_MAP[policy.severity] || 'badge-muted'}`}>
              {policy.severity}
            </span>
            {policy.jurisdiction && policy.jurisdiction !== 'GLOBAL' && (
              <span className="badge badge-muted">{policy.jurisdiction}</span>
            )}
            <span style={{
              marginLeft: 'auto', fontSize: 10.5, fontWeight: 600,
              color: policy.enabled ? 'var(--green)' : 'var(--text-muted)',
              background: policy.enabled ? 'var(--green-light)' : 'var(--bg-card)',
              border: `1px solid ${policy.enabled ? 'rgba(5,150,105,.15)' : 'var(--border)'}`,
              padding: '2px 8px', borderRadius: 20, whiteSpace: 'nowrap', flexShrink: 0,
            }}>
              {policy.enabled ? '● Active' : '○ Inactive'}
            </span>
          </div>

          <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.65, margin: '0 0 6px' }}>
            {policy.description}
          </p>

          {/* Expanded detail */}
          {(policy.regulation || policy.source || policy.rationale) && (
            <div>
              <button
                onClick={() => setExpanded(e => !e)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontSize: 10.5, color: 'var(--text-muted)', padding: 0,
                  display: 'flex', alignItems: 'center', gap: 4,
                }}
              >
                <Eye size={10} /> {expanded ? 'Hide' : 'Show'} legal reference
              </button>
              {expanded && (
                <div style={{
                  marginTop: 8, padding: '8px 11px',
                  background: 'var(--bg-card)', borderRadius: 'var(--radius-xs)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: 11, lineHeight: 1.65, color: 'var(--text-muted)',
                }}>
                  {policy.regulation && (
                    <div><span style={{ fontWeight: 600, color: 'var(--text-dim)' }}>Regulation: </span>{policy.regulation}</div>
                  )}
                  {policy.source && (
                    <div style={{ marginTop: 2 }}><span style={{ fontWeight: 600, color: 'var(--text-dim)' }}>Source: </span>{policy.source}</div>
                  )}
                  {policy.rationale && (
                    <div style={{ marginTop: 4, fontStyle: 'italic', color: 'var(--text-muted)' }}>{policy.rationale}</div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Delete (custom policies only) */}
        {canDelete && !baseline && !principle && (
          <button
            onClick={() => onDelete(policy)}
            disabled={isDeletingThis}
            className="btn btn-danger btn-xs"
            style={{ flexShrink: 0 }}
          >
            {isDeletingThis
              ? <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
              : <Trash2 size={12} />
            }
          </button>
        )}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   SECTION HEADER
───────────────────────────────────────────────────────────── */
function SectionHeader({ icon: Icon, title, subtitle, count, color, bg }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '11px 16px',
      background: bg,
      borderRadius: 'var(--radius-sm)',
      border: `1px solid ${color}22`,
      marginBottom: 8,
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: 8,
        background: `${color}18`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>
        <Icon size={15} style={{ color }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)' }}>{title}</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>{subtitle}</div>
      </div>
      <span style={{
        fontSize: 11, fontWeight: 700, color,
        background: `${color}14`, border: `1px solid ${color}22`,
        padding: '2px 9px', borderRadius: 20,
      }}>{count} active</span>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   MAIN PAGE
───────────────────────────────────────────────────────────── */
export default function PoliciesPage() {
  const { user, hasPermission } = useAuth()
  const isAdmin = user?.role === 'super_admin'
  const canWrite = isAdmin
  const canDelete = isAdmin
  const canToggle = isAdmin

  /* Remote custom policies from backend */
  const [customPolicies, setCustomPolicies] = useState([])
  const [loading, setLoading] = useState(false)

  /* UI state */
  const [activeTab, setActiveTab] = useState('baseline')  // 'baseline' | 'principle' | 'custom'
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(1)
  const [showModal, setShowModal] = useState(false)
  const [jsonMode, setJsonMode] = useState(false)
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE_JSON, null, 2))
  const [jsonError, setJsonError] = useState('')
  const [deleting, setDeleting] = useState(null)
  const [toggling, setToggling] = useState(null)
  const [form, setForm] = useState({
    name: '', description: '', policy_type: 'fairness', severity: 'medium', jurisdiction: 'IN',
  })
  const [principleSubTab, setPrincipleSubTab] = useState('all') // 'all' | 'law' | 'forward'

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const r = await policiesAPI.list()
      const remote = Array.isArray(r.data) ? r.data : []
      // Only keep remote policies that are truly custom (not in our static sets)
      const staticIds = new Set([...BASELINE_POLICIES, ...PRINCIPLE_POLICIES].map(p => p.id))
      setCustomPolicies(remote.filter(p => !staticIds.has(p.id) && !isBaseline(p) && !isPrinciple(p)))
    } catch {
      setCustomPolicies([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    window.addEventListener('kavachx:simulation-complete', load)
    return () => window.removeEventListener('kavachx:simulation-complete', load)
  }, [load])
  useEffect(() => { setPage(1) }, [activeTab, filter, principleSubTab])

  /* Filter logic per tab */
  const baselineFiltered = filter === 'all'
    ? BASELINE_POLICIES
    : BASELINE_POLICIES.filter(p => p.policy_type === filter || p.jurisdiction === filter)

  const principleFiltered = (() => {
    let src = PRINCIPLE_POLICIES
    if (principleSubTab === 'law') src = src.filter(p => p.tier === 1)
    if (principleSubTab === 'forward') src = src.filter(p => p.tier === 2)
    if (filter !== 'all') src = src.filter(p => p.policy_type === filter || p.jurisdiction === filter)
    return src
  })()

  const customFiltered = filter === 'all'
    ? customPolicies
    : customPolicies.filter(p => p.policy_type === filter)

  const currentList = activeTab === 'baseline' ? baselineFiltered
    : activeTab === 'principle' ? principleFiltered
      : customFiltered

  const paginated = currentList.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  /* Toggle (custom only — baseline/principle are always on) */
  const toggle = async (policy) => {
    if (!canToggle || isBaseline(policy) || isPrinciple(policy)) return
    setToggling(policy.id)
    setCustomPolicies(prev => prev.map(p => p.id === policy.id ? { ...p, enabled: !p.enabled } : p))
    try { await policiesAPI.toggle(policy.id, !policy.enabled) }
    catch { load() }
    finally { setToggling(null) }
  }

  const remove = async (policy) => {
    if (!canDelete || isBaseline(policy) || isPrinciple(policy)) return
    if (!window.confirm(`Delete policy "${policy.name}"? This cannot be undone.`)) return
    setDeleting(policy.id)
    try { await policiesAPI.delete(policy.id) } catch { }
    finally {
      setDeleting(null)
      setCustomPolicies(prev => prev.filter(p => p.id !== policy.id))
    }
  }

  const create = async () => {
    let payload
    if (jsonMode) {
      try {
        payload = JSON.parse(jsonInput)
        if (!payload.name?.trim()) { setJsonError('Policy name is required.'); return }
        if (!payload.policy_type) { setJsonError('policy_type is required.'); return }
        setJsonError('')
      } catch (e) { setJsonError(`Invalid JSON: ${e.message}`); return }
    } else {
      if (!form.name.trim()) return
      payload = { ...form, rules: [{ field: 'confidence', operator: 'lt', value: 0.6, action: 'ALERT' }] }
    }
    try {
      await policiesAPI.create(payload)
      setShowModal(false); resetForm(); load()
    } catch {
      const local = { ...payload, id: `local-${Date.now()}`, enabled: true }
      setCustomPolicies(prev => [local, ...prev])
      setShowModal(false); resetForm()
    }
  }

  const resetForm = () => {
    setForm({ name: '', description: '', policy_type: 'fairness', severity: 'medium', jurisdiction: 'IN' })
    setJsonInput(JSON.stringify(SAMPLE_JSON, null, 2))
    setJsonError(''); setJsonMode(false)
  }

  const handleFileUpload = (e) => {
    const file = e.target.files[0]; if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => { setJsonInput(ev.target.result); setJsonMode(true) }
    reader.readAsText(file); e.target.value = ''
  }

  /* Tab counts */
  const tabCounts = {
    baseline: BASELINE_POLICIES.filter(p => p.enabled).length,
    principle: PRINCIPLE_POLICIES.filter(p => p.enabled).length,
    custom: customPolicies.filter(p => p.enabled).length,
  }

  const FILTER_OPTIONS = ['all', ...POLICY_TYPES, 'IN', 'GLOBAL']

  return (
    <div>
      {/* ── Page header ── */}
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
          <div>
            <div className="page-eyebrow">Governance</div>
            <h1 className="page-title">Governance Policies</h1>
            <p className="page-desc">
              Three-layer governance architecture: Baseline (regulatory baseline), Principle (legal derivation + forward-looking), and Custom policies.
              {canToggle && <span style={{ color: 'var(--accent-2)' }}> Super Admin active.</span>}
            </p>
          </div>
          {canWrite && (
            <button className="btn btn-primary" onClick={() => setShowModal(true)}>
              <Plus size={14} /> New Policy
            </button>
          )}
        </div>
      </div>

      {/* ── Summary tiles ── */}
      <div className="stats-row" style={{ marginBottom: 20 }}>
        <div className="stat-card" style={{ '--stat-color': 'var(--accent)', '--stat-bg': 'var(--accent-light)' }}>
          <div className="stat-icon"><Shield size={18} /></div>
          <div className="stat-value">{BASELINE_POLICIES.length}</div>
          <div className="stat-label">Baseline Policies</div>
        </div>
        <div className="stat-card" style={{ '--stat-color': 'var(--cyan)', '--stat-bg': 'var(--cyan-light)' }}>
          <div className="stat-icon"><BookOpen size={18} /></div>
          <div className="stat-value">{PRINCIPLE_POLICIES.filter(p => p.tier === 1).length}</div>
          <div className="stat-label">Law-Derived Principles</div>
        </div>
        <div className="stat-card" style={{ '--stat-color': 'var(--purple)', '--stat-bg': 'var(--purple-light)' }}>
          <div className="stat-icon"><Lightbulb size={18} /></div>
          <div className="stat-value">{PRINCIPLE_POLICIES.filter(p => p.tier === 2).length}</div>
          <div className="stat-label">Forward-Looking Principles</div>
        </div>
        <div className="stat-card" style={{ '--stat-color': 'var(--green)', '--stat-bg': 'var(--green-light)' }}>
          <div className="stat-icon"><Plus size={18} /></div>
          <div className="stat-value">{customPolicies.length}</div>
          <div className="stat-label">Custom Policies</div>
        </div>
      </div>

      {/* ── Layer tabs ── */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border)', marginBottom: 16 }}>
        {[
          { key: 'baseline', label: 'Baseline', icon: Shield, count: tabCounts.baseline },
          { key: 'principle', label: 'Principle', icon: BookOpen, count: tabCounts.principle },
          { key: 'custom', label: 'Custom', icon: Plus, count: tabCounts.custom },
        ].map(({ key, label, icon: Icon, count }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '9px 16px',
              border: 'none', borderBottom: `2px solid ${activeTab === key ? 'var(--accent)' : 'transparent'}`,
              background: 'none', cursor: 'pointer',
              fontSize: 12.5, fontWeight: activeTab === key ? 700 : 500,
              color: activeTab === key ? 'var(--accent)' : 'var(--text-muted)',
              marginBottom: -1, transition: 'all .13s',
            }}
          >
            <Icon size={13} />
            {label}
            <span style={{
              fontSize: 10, fontWeight: 700,
              background: activeTab === key ? 'var(--accent)' : 'var(--bg-elevated)',
              color: activeTab === key ? '#fff' : 'var(--text-muted)',
              padding: '1px 6px', borderRadius: 10, marginLeft: 2,
            }}>{count}</span>
          </button>
        ))}
      </div>

      {/* ── Section header descriptions ── */}
      {activeTab === 'baseline' && (
        <SectionHeader
          icon={Shield}
          title="Baseline Policy Layer"
          subtitle="Mandatory regulatory compliance — DPDP Act 2023, RBI, SEBI, IRDAI, MeitY, IT Act, NHA/ABDM, and KavachX India contextual standards. Cannot be disabled."
          count={BASELINE_POLICIES.filter(p => p.enabled).length}
          color="var(--accent)"
          bg="var(--accent-light)"
        />
      )}
      {activeTab === 'principle' && (
        <>
          <SectionHeader
            icon={BookOpen}
            title="Principle Policy Layer"
            subtitle="Governance rules not yet operationalised by existing technical standards — derived from Indian law and regulatory guidance, plus forward-looking principles for emerging AI risks."
            count={PRINCIPLE_POLICIES.filter(p => p.enabled).length}
            color="var(--cyan)"
            bg="var(--cyan-light)"
          />
          {/* Principle sub-tabs */}
          <div style={{ display: 'flex', gap: 5, marginBottom: 12 }}>
            {[
              { key: 'all', label: `All (${PRINCIPLE_POLICIES.length})` },
              { key: 'law', label: `Law-Derived (${PRINCIPLE_POLICIES.filter(p => p.tier === 1).length})` },
              { key: 'forward', label: `Forward-Looking (${PRINCIPLE_POLICIES.filter(p => p.tier === 2).length})` },
            ].map(({ key, label }) => (
              <button
                key={key}
                className={`filter-pill ${principleSubTab === key ? 'active' : ''}`}
                onClick={() => setPrincipleSubTab(key)}
              >{label}</button>
            ))}
          </div>
          {principleSubTab === 'law' && (
            <div className="alert alert-info" style={{ marginBottom: 12 }}>
              <BookOpen size={14} style={{ flexShrink: 0 }} />
              <span>These principles are derived from Indian laws and regulatory guidance — including the Constitution, DPDP Act, Consumer Protection Act, RTI Act, Code on Social Security, and digital infrastructure policy — where no specific technical governance standard currently exists at the model level.</span>
            </div>
          )}
          {principleSubTab === 'forward' && (
            <div className="alert" style={{ background: 'var(--purple-light)', borderColor: 'rgba(124,58,237,.15)', color: 'var(--purple)', marginBottom: 12 }}>
              <Lightbulb size={14} style={{ flexShrink: 0 }} />
              <span>Forward-looking principles address governance gaps not covered by any current industry standard or commercial platform. These include adversarial robustness, causal fairness classification, cryptographic audit integrity, citizen redress, inclusion auditing, and longitudinal fairness tracking.</span>
            </div>
          )}
        </>
      )}
      {activeTab === 'custom' && (
        <SectionHeader
          icon={Plus}
          title="Custom Policy Layer"
          subtitle="Organisation-defined policies created by Super Admins. Can be enabled, disabled, or deleted. Full JSON authoring supported."
          count={customPolicies.filter(p => p.enabled).length}
          color="var(--green)"
          bg="var(--green-light)"
        />
      )}

      {/* ── Filter pills ── */}
      <div className="filter-pills mb-16">
        {FILTER_OPTIONS.map(f => (
          <button
            key={f}
            className={`filter-pill ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All types' : f.replace('_', ' ')}
          </button>
        ))}
      </div>

      {/* ── Role notice ── */}
      {!canWrite && activeTab === 'custom' && (
        <div className="alert alert-warn mb-16">
          <Info size={14} style={{ flexShrink: 0 }} />
          <span>Only Super Admins can create custom policies. Your current role is <strong>{user?.role_label || user?.role}</strong>.</span>
        </div>
      )}

      {/* ── Policy list ── */}
      {loading && activeTab === 'custom' ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}><div className="spinner" /></div>
      ) : currentList.length === 0 ? (
        <div className="empty card">
          <Shield size={32} className="empty-icon" />
          <div className="empty-title">
            {activeTab === 'custom' ? 'No custom policies yet' : 'No policies match filter'}
          </div>
          <div className="empty-desc">
            {activeTab === 'custom' && canWrite
              ? 'Create your first custom policy using the New Policy button above.'
              : 'Adjust the filter to see policies.'}
          </div>
        </div>
      ) : (
        <div className="card">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {paginated.map(policy => (
              <PolicyCard
                key={policy.id}
                policy={policy}
                canToggle={canToggle}
                canDelete={canDelete}
                onToggle={toggle}
                onDelete={remove}
                toggling={toggling}
                deleting={deleting}
              />
            ))}
          </div>
          <Pagination page={page} total={currentList.length} pageSize={PAGE_SIZE} onChange={setPage} />
        </div>
      )}

      {/* ── Create modal — super_admin only ── */}
      {showModal && canWrite && (
        <div className="modal-overlay" onClick={e => e.target === e.currentTarget && (setShowModal(false), resetForm())}>
          <div className="modal modal-lg">
            <div className="modal-header">
              <span className="modal-title">Create Custom Policy</span>
              <div style={{ display: 'flex', gap: 8 }}>
                <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                  <button className={`btn btn-xs ${!jsonMode ? 'btn-primary' : 'btn-ghost'}`} style={{ borderRadius: 0 }} onClick={() => setJsonMode(false)}>Form</button>
                  <button className={`btn btn-xs ${jsonMode ? 'btn-primary' : 'btn-ghost'}`} style={{ borderRadius: 0 }} onClick={() => setJsonMode(true)}><Code size={12} /> JSON</button>
                </div>
                <button className="btn btn-ghost btn-xs" onClick={() => { setShowModal(false); resetForm() }}><X size={14} /></button>
              </div>
            </div>

            <div className="modal-body">
              {!jsonMode ? (
                <>
                  <div className="form-group">
                    <label className="form-label">Policy Name *</label>
                    <input className="form-input" placeholder="e.g. Credit Score Threshold"
                      value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Description</label>
                    <textarea className="form-input" rows={3} placeholder="What does this policy enforce?"
                      value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))} />
                  </div>
                  <div className="grid-2" style={{ gap: 12 }}>
                    <div className="form-group">
                      <label className="form-label">Type</label>
                      <select className="form-input" value={form.policy_type} onChange={e => setForm(f => ({ ...f, policy_type: e.target.value }))}>
                        {POLICY_TYPES.map(t => <option key={t} value={t}>{t.replace('_', ' ')}</option>)}
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Severity</label>
                      <select className="form-input" value={form.severity} onChange={e => setForm(f => ({ ...f, severity: e.target.value }))}>
                        {SEVERITIES.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </div>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Jurisdiction</label>
                    <select className="form-input" value={form.jurisdiction} onChange={e => setForm(f => ({ ...f, jurisdiction: e.target.value }))}>
                      {JURISDICTIONS.map(j => <option key={j} value={j}>{j}</option>)}
                    </select>
                  </div>
                </>
              ) : (
                <>
                  <div className="alert alert-info" style={{ marginBottom: 0 }}>
                    <FileJson size={14} style={{ flexShrink: 0 }} />
                    <span>Paste or upload a JSON policy file. The sample shows the required structure.</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
                      <Upload size={13} /> Upload JSON File
                      <input type="file" accept=".json,application/json" style={{ display: 'none' }} onChange={handleFileUpload} />
                    </label>
                    <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>or edit the template below</span>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Policy JSON</label>
                    <textarea
                      className="form-input"
                      style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, minHeight: 220, lineHeight: 1.7 }}
                      value={jsonInput}
                      onChange={e => { setJsonInput(e.target.value); setJsonError('') }}
                      spellCheck={false}
                    />
                  </div>
                  {jsonError && (
                    <div className="alert alert-error"><Info size={14} style={{ flexShrink: 0 }} /><span>{jsonError}</span></div>
                  )}
                  <details style={{ fontSize: 12 }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 600, color: 'var(--text-dim)', marginBottom: 8 }}>JSON Schema Reference</summary>
                    <div className="code-block" style={{ fontSize: 11 }}>{`{
  "name": "string (required)",
  "description": "string",
  "policy_type": "fairness | safety | compliance | performance | llm_safety",
  "severity": "low | medium | high | critical",
  "jurisdiction": "GLOBAL | IN | EU | US | APAC",
  "rules": [{ "field": "confidence|risk_score|toxicity_score", "operator": "lt|gt|lte|gte|eq", "value": 0.0–1.0, "action": "ALERT|BLOCK|HUMAN_REVIEW" }]
}`}</div>
                  </details>
                </>
              )}
            </div>

            <div className="modal-footer">
              <button className="btn btn-secondary btn-sm" onClick={() => { setShowModal(false); resetForm() }}>Cancel</button>
              <button className="btn btn-primary btn-sm" onClick={create} disabled={jsonMode ? !jsonInput.trim() : !form.name.trim()}>
                <Plus size={13} /> Create Policy
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
