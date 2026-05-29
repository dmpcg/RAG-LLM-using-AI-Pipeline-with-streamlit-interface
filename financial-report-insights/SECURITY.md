# Security Policy & Hardening Status
## RAG-LLM Financial Report Insights

**Last Updated:** 2026-05-29
**Status:** Hardening Complete, Vulnerabilities Remediated
**Audit Date:** 2026-02-22

---

## Executive Summary

This document outlines the security posture, remediation status, and hardening measures implemented in the RAG-LLM Financial Report Insights project. A comprehensive security audit identified 10 vulnerabilities (3 CRITICAL, 7 HIGH/MEDIUM) in February 2026. All critical and high-priority items have been remediated and verified.

**Current Risk Level:** MEDIUM/LOW (after remediation)

---

## Supported Versions

| Version | Status | Support Level | Security Updates |
|---------|--------|---------------|------------------|
| Current (2026) | Active | Full Support | Yes, continuous |
| Previous (2025) | EOL | Limited | Security patches only |

---

## Security Hardening Measures

### Credential Management
- **Neo4j Password Rotation:** Implemented with 1Password vault integration
- **Git History Cleanup:** Removed .env from all historical commits using BFG Repo-Cleaner
- **Secrets Management:** AWS Secrets Manager (production) / 1Password (development)
- **No Hardcoded Credentials:** .gitignore protects against accidental commits

### Dependency Security
- **Pin Policy:** All dependencies pinned to exact versions in requirements.lock
- **CVE Monitoring:** Automated scanning via Safety/Dependabot
- **Update Process:** Weekly security updates, monthly major version review
- **32 Known CVEs Fixed:** All critical/high severity addressed via pinned versions

**Key Pinned Versions:**
```
fastapi==0.115.6
starlette==0.41.3
uvicorn==0.34.0
httpx==0.28.1
streamlit==1.43.2
PyMuPDF==1.25.3
aiohttp==3.11.11
requests==2.32.3
python-multipart==0.0.12
certifi==2024.12.14
neo4j==5.14.0
```

### API Security

#### Rate Limiting
- **Per-Endpoint Limits:** Prevents DoS and cost explosion
  - `/health`: 1000 req/min (monitoring exempt)
  - `/query`: 10 req/min (LLM-backed)
  - `/analyze`: 5 req/min (expensive operations)
  - `/export`: 3 req/min (resource-intensive)
- **Response Headers:** X-RateLimit-Remaining, Retry-After (429 status)
- **Audit Logging:** All violations logged with client IP and timestamp

#### File Upload Validation
- **Whitelist Extensions:** .pdf, .docx, .xlsx, .xls, .txt only
- **Max Size:** 50 MB (reduced from 200 MB)
- **MIME Type Validation:** File type verification on upload
- **Path Traversal Prevention:** Filenames sanitized, no directory traversal
- **Security Logging:** All uploads logged with file hash

#### Input Validation
- **Field Name Validation:** Alphanumeric and underscore only
- **Field Length Limits:** Max 100 characters per field name
- **Numeric Range Checks:** NaN and Infinity prevented
- **Request Size Limit:** 1 MB max Content-Length
- **Field Count Cap:** Max 200 fields per request via Pydantic validators

### CORS Configuration
- **Explicit Origins:** No wildcard (*) in production
- **Credentials:** Disabled by default (can enable per deployment)
- **Preflight Caching:** max_age 3600 seconds
- **Methods:** POST, GET only (no DELETE, PUT in API)
- **Headers:** Content-Type, Authorization, Accept

### Logging & Audit Trail
- **Structured JSON Logging:** All sensitive operations logged
- **Sensitive Endpoints Tracked:** /analyze, /export, /documents, /upload
- **Rate Limit Violations:** Logged with rate limit details
- **Error Logging:** 4xx/5xx with sanitized error messages (no stack traces)
- **Password Redaction:** Neo4j URI/password stripped from all logs

### Docker Security Hardening
- **Capability Dropping:** cap_drop ALL (rag-app only adds NET_BIND_SERVICE)
- **Privilege Escaping:** security_opt no-new-privileges:true
- **Read-Only Filesystem:** read_only true with tmpfs for /tmp and .cache
- **Resource Limits:** CPU capped at 2-4 cores (rag-app), 1-2 cores (neo4j)
- **Non-Root User:** Container runs as unprivileged user
- **Health Checks:** Automatic restart on failure

### Security Headers
- **X-Content-Type-Options:** nosniff (prevent MIME-type sniffing)
- **X-Frame-Options:** DENY (prevent clickjacking)
- **Content-Security-Policy:** Strict (hardcoded CSS only, no eval)
- **Permissions-Policy:** Restrict browser features
- **Pragma:** no-cache (cache control)
- **Expires:** Set to past (disable caching)

### Code-Level Hardening
- **AST-Based Formula Evaluation:** Safe eval() replacement (no exec/compile)
- **Cypher Injection Prevention:** Regex allowlist for graph index names
- **SQL-Like Injection:** Parameterized all Neo4j queries (UNWIND batching)
- **SSRF Prevention:** OLLAMA_HOST restricted to localhost/docker internal
- **Thread-Safe Singleton:** _get_rag() guarded with threading.Lock
- **Safe Division:** safe_divide(num, denom, default=None) for all divisions

### Continuous Integration / Deployment
- **Pre-Commit Hooks:** Detect secrets before commit
- **Dependabot Integration:** Weekly dependency update PRs
- **Security Scanning:** Bandit (code), Safety (deps), Trivy (containers)
- **SHA-Pinned GitHub Actions:** Prevents supply chain attacks
- **Artifact Signing:** (planned for production)

---

## Remediation Status

### Phase 1: Credential Response (COMPLETED ✅)
- [x] Neo4j password rotated (2026-03-16)
- [x] Git history cleaned (removed .env via BFG)
- [x] .gitignore committed (prevents future secret commits)
- [x] .env.example sanitized (no credentials, guidance added)

### Phase 2: API & Infrastructure Security (COMPLETED ✅)
- [x] Per-endpoint rate limiting implemented (code in api.py)
- [x] File upload validation added (whitelist, max size, MIME check)
- [x] Input validation hardened (field names, ranges, NaN/Inf)
- [x] Dependency versions pinned (requirements.lock)
- [x] Docker security hardening (cap_drop, read_only, tmpfs)
- [x] Audit logging on sensitive endpoints

### Phase 3: Secrets & Compliance (COMPLETED ✅)
- [x] Secrets management via 1Password/AWS
- [x] CORS hardening (explicit origins, no credentials)
- [x] Database connection security (TLS, timeouts, pooling)
- [x] Credential rotation procedures (30-90 day schedule)

### Phase 4: Continuous Improvement (COMPLETED ✅)
- [x] CI/CD security scanning (Bandit, Safety, Trivy)
- [x] Request signing for sensitive endpoints (HMAC-SHA256)
- [x] Rate limit monitoring (Prometheus metrics, Grafana dashboard)
- [x] Incident response procedures documented

---

## Compliance & Standards

### Frameworks Aligned
- **OWASP Top 10:** Addresses A01-A10 (injection, auth, data exposure, etc.)
- **CIS Benchmarks:** Docker, Kubernetes, and application security
- **NIST Cybersecurity Framework:** Identify, Protect, Detect, Respond, Recover
- **GDPR:** Data protection, privacy by design, audit trails

### Certification Status
| Standard | Current | Target |
|----------|---------|--------|
| SOC2 Type II | Ready for audit | Q3 2026 |
| ISO27001 | Ready for audit | Q3 2026 |
| GDPR | Compliant | Ongoing |

---

## Incident Response

### Reporting Security Issues
**Do not open public GitHub issues for security vulnerabilities.**

Email: devmcgrath@gmail.com

**Response Timeline:**
- Critical: 24 hours initial response
- High: 48 hours initial response
- Medium: 1 week initial response

### Remediation Process
1. Vulnerability assessment and severity rating
2. Root cause analysis
3. Patch development and testing
4. Staged rollout (dev → staging → production)
5. Post-incident review and documentation

---

## Monitoring & Alerts

### Real-Time Monitoring
- **Rate Limit Violations:** Alert on >100 violations/hour per endpoint
- **Failed Authentication:** Alert on >10 failures/hour per user
- **Dependency Vulnerabilities:** Automatic PR on detection
- **Container Image Scan:** Weekly Trivy scan on latest build

### Audit Log Retention
- **Production:** 90 days (GDPR/SOC2 requirement)
- **Staging:** 30 days
- **Development:** 7 days

---

## Update Policy

### Security Patches
- **Critical:** Applied within 24 hours of availability
- **High:** Applied within 1 week
- **Medium:** Applied during next regular update cycle
- **Low:** Batched with feature releases

### Major Version Updates
- **Quarterly Review:** All major versions reviewed
- **Staged Rollout:** Dev → Staging → Production
- **Rollback Plan:** Previous version always available in prod

---

## Additional Security Resources

For detailed information, see:
- [docs/security/AUDIT.md](docs/security/AUDIT.md) — Complete audit findings (60+ pages)
- [docs/security/REMEDIATION_PLAN.md](docs/security/REMEDIATION_PLAN.md) — Phase-by-phase remediation roadmap
- [docs/security/HARDENING_CHECKLIST.md](docs/security/HARDENING_CHECKLIST.md) — Quick reference checklist
- [docs/security/CVE_STATUS.md](docs/security/CVE_STATUS.md) — CVE remediation tracking

---

## Key Contacts

| Role | Owner |
|------|-------|
| Security Lead | devmcgrath@gmail.com |
| DevOps / Infrastructure | See project maintainers |
| Incident Response | See project maintainers |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-29 | 1.0 | Consolidated security documentation, all critical/high items remediated |
| 2026-03-16 | 0.9 | P0+P1 remediation complete (32 CVEs addressed) |
| 2026-02-22 | 0.1 | Initial security audit completed |

---

**Last Review:** 2026-05-29
**Next Review:** 2026-08-29 (quarterly)
**Status:** HARDENED & COMPLIANT
