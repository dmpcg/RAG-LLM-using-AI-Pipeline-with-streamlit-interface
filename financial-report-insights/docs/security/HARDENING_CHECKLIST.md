# Security Hardening Quick Reference Checklist
## RAG-LLM Financial Report Insights

**Purpose:** Quick reference for developers and operators
**Status:** All items completed as of 2026-03-16
**Update Frequency:** Quarterly

---

## Pre-Deployment Security Checklist

Use this before any deployment to production or staging.

### Code Security (15 minutes)
- [ ] Run Bandit: `bandit -r . -ll` — 0 HIGH/MEDIUM findings
- [ ] Run Safety: `safety check` — No known CVEs
- [ ] Semgrep: `semgrep --config=p/security-audit .` — No vulnerabilities
- [ ] Code review completed by 2+ people
- [ ] No secrets in code (check git diff)
- [ ] No debug statements or print() logging

### Dependency Security (10 minutes)
- [ ] Using requirements.lock (not loose ranges)
- [ ] All major versions pinned to patch versions
- [ ] No dev dependencies in production
- [ ] No deprecated packages
- [ ] Safety check passes with no CVEs

### Configuration Security (10 minutes)
- [ ] CORS origins do NOT include wildcard (*)
- [ ] CORS origins match deployment target
- [ ] Rate limit endpoints configured per deployment
- [ ] Secrets sourced from vault, not .env
- [ ] Database connection uses TLS
- [ ] Logging redacts passwords and tokens

### Docker Security (10 minutes)
- [ ] Docker image builds without warnings
- [ ] Trivy scan: 0 CRITICAL findings
- [ ] Non-root user in Dockerfile
- [ ] cap_drop ALL in docker-compose.yml
- [ ] read_only: true enabled
- [ ] Resource limits set (CPU, memory)

### API Security (10 minutes)
- [ ] Rate limiting enabled on all endpoints
- [ ] File upload validation active
- [ ] Input validation hardened
- [ ] Authentication required for sensitive endpoints
- [ ] Security headers present (CSP, X-Frame-Options, etc.)
- [ ] HTTPS/TLS enabled (no plain HTTP)

### Audit & Logging (5 minutes)
- [ ] Structured JSON logging enabled
- [ ] Sensitive endpoints logged
- [ ] Log retention policy set
- [ ] Log rotation configured
- [ ] No personal data in logs

---

## Weekly Security Review Checklist

### Dependency Updates (30 minutes)
- [ ] Review Dependabot PRs (if available)
- [ ] Check Safety for new CVEs
- [ ] Merge non-breaking updates
- [ ] Schedule major version updates for next sprint

### Monitoring & Alerts (15 minutes)
- [ ] Review rate limit violation logs
- [ ] Check for authentication failures
- [ ] Verify no unusual API patterns
- [ ] Confirm health checks passing

### Access Review (15 minutes)
- [ ] Review team members with repo access
- [ ] Remove access for inactive team members
- [ ] Verify secrets not exposed in recent commits
- [ ] Check for accidental .env commits (should be prevented by .gitignore)

---

## Monthly Security Review Checklist

### Vulnerability Management (1 hour)
- [ ] Run all security scans (Bandit, Safety, Semgrep, Trivy)
- [ ] Review and prioritize findings
- [ ] Create GitHub issues for HIGH/CRITICAL findings
- [ ] Assign owners to remediation tasks
- [ ] Schedule fixes for next sprint

### Dependency Audit (30 minutes)
- [ ] Check for outdated major versions
- [ ] Plan major version upgrades
- [ ] Review deprecation notices
- [ ] Update CHANGELOG with dependency changes

### Access & Secrets Review (30 minutes)
- [ ] Review database password rotation (should be 30-90 days)
- [ ] Check API key rotation schedule
- [ ] Verify no hardcoded secrets in code
- [ ] Audit git history for accidental commits (git log --all -S "password")

### Compliance Check (30 minutes)
- [ ] Verify SOC2/ISO27001 requirements met
- [ ] Check audit logging completeness
- [ ] Review data retention policies
- [ ] Confirm encryption at rest/in transit

---

## Quarterly Security Audit Checklist

### Full Security Assessment (4-6 hours)
- [ ] Review all findings from previous audit
- [ ] Verify remediations are still in place
- [ ] Run comprehensive security scans
- [ ] Analyze attack logs for trends
- [ ] Update risk assessment matrix

### Architecture Review (2-3 hours)
- [ ] Verify network isolation still effective
- [ ] Review Docker security options
- [ ] Check database connection security
- [ ] Assess service-to-service communication

### Compliance Verification (1-2 hours)
- [ ] SOC2 controls still implemented
- [ ] ISO27001 requirements met
- [ ] GDPR data protection verified
- [ ] Audit trail completeness checked

### Team Training (1 hour)
- [ ] Review security policies with team
- [ ] Update runbooks if needed
- [ ] Train new team members on security practices
- [ ] Discuss lessons learned from month

---

## Rate Limiting Configuration Checklist

### Per-Endpoint Limits (Verify Active)
- [ ] `/health`: 1000 req/min (monitoring)
- [ ] `/query`: 10 req/min (LLM calls)
- [ ] `/analyze`: 5 req/min (expensive analysis)
- [ ] `/export`: 3 req/min (resource intensive)
- [ ] `/documents`: 20 req/min (file operations)

### Rate Limit Verification Test
```bash
# Test /query endpoint rate limiting (should hit limit at ~10 req/min)
for i in {1..20}; do
  curl -s -X POST http://localhost:8504/query \
    -H "Content-Type: application/json" \
    -d '{"text":"test","top_k":3}' \
    -w "%{http_code}\n"
done | sort | uniq -c

# Expected: 10x 200s, 10x 429s
```

---

## File Upload Security Checklist

### Validation Rules (Verify Enforced)
- [ ] Extensions: .pdf, .docx, .xlsx, .xls, .txt only
- [ ] Max size: 50 MB
- [ ] MIME type validated
- [ ] Filenames sanitized (no path traversal)
- [ ] File hash logged for audit trail

### Test Upload Security
```bash
# Test valid file (should succeed)
curl -F "file=@report.pdf" http://localhost:8504/documents/upload
# Expected: 200 OK

# Test invalid type (should fail)
curl -F "file=@virus.exe" http://localhost:8504/documents/upload
# Expected: 400 Bad Request

# Test size limit (should fail)
# Create 100MB file: dd if=/dev/zero of=large.txt bs=1M count=100
curl -F "file=@large.txt" http://localhost:8504/documents/upload
# Expected: 413 Payload Too Large
```

---

## Input Validation Checklist

### Validation Rules (Verify Active)
- [ ] Field names: alphanumeric + underscore only
- [ ] Field name length: max 100 characters
- [ ] Field count: max 200 per request
- [ ] Numeric values: NaN/Infinity rejected
- [ ] Request size: 1 MB max

### Test Input Validation
```bash
# Valid request
curl -X POST http://localhost:8504/analyze \
  -H "Content-Type: application/json" \
  -d '{"revenue":1000000,"expenses":500000}'
# Expected: 200 OK

# Invalid field name (with dash)
curl -X POST http://localhost:8504/analyze \
  -H "Content-Type: application/json" \
  -d '{"invalid-field":100}'
# Expected: 400 Bad Request

# NaN value
curl -X POST http://localhost:8504/analyze \
  -H "Content-Type: application/json" \
  -d '{"revenue":NaN}'
# Expected: 400 Bad Request
```

---

## Secrets Management Checklist

### Development Environment
- [ ] 1Password vault configured
- [ ] NEO4J_PASSWORD sourced from vault
- [ ] No .env checked into git
- [ ] Team trained on secret rotation

### Production Environment
- [ ] AWS Secrets Manager configured
- [ ] Application reads secrets on startup
- [ ] Secrets not logged or printed
- [ ] Credential rotation automated (30-90 day schedule)
- [ ] Failed rotation alerts set up

### Verification
```bash
# Verify no secrets in recent commits
git log -p --all -S "NEO4J_PASSWORD" | head -50
# Should return nothing (password removed from history)

# Check current .env is in .gitignore
cat .gitignore | grep "^\.env"
# Should print: .env
```

---

## Docker Security Verification Checklist

### Security Options (Verify with docker inspect)
- [ ] cap_drop: ALL
- [ ] cap_add: NET_BIND_SERVICE (rag-app only)
- [ ] security_opt: no-new-privileges:true
- [ ] read_only: true
- [ ] tmpfs: /tmp, /app/.cache

### Resource Limits (Verify with docker inspect)
- [ ] rag-app: CPU 2-4 cores
- [ ] rag-app: Memory 1-2 GB
- [ ] neo4j: CPU 1-2 cores
- [ ] neo4j: Memory 2-4 GB

### Verification Command
```bash
docker inspect rag-financial-insights | \
  jq '.[] | {CapDrop, CapAdd, SecurityOpt, ReadonlyRootfs, HostConfig: {CpuQuota, Memory}}'

# Should show:
# {
#   "CapDrop": ["ALL"],
#   "CapAdd": ["NET_BIND_SERVICE"],
#   "SecurityOpt": ["no-new-privileges:true"],
#   "ReadonlyRootfs": true,
#   "HostConfig": { "CpuQuota": "...", "Memory": 1073741824 }
# }
```

---

## CORS Security Verification Checklist

### Configuration Checks
- [ ] No wildcard (*) in CORS origins
- [ ] Origins match deployment target
- [ ] Credentials disabled by default
- [ ] Max age set to 3600 seconds
- [ ] Methods limited to GET, POST

### Test CORS Enforcement
```bash
# Test allowed origin (should include header)
curl -H "Origin: http://localhost:8501" \
     -H "Access-Control-Request-Method: POST" \
     -v http://localhost:8504/query
# Should see: Access-Control-Allow-Origin: http://localhost:8501

# Test blocked origin (should NOT include header)
curl -H "Origin: http://attacker.com" \
     -H "Access-Control-Request-Method: POST" \
     -v http://localhost:8504/query
# Should NOT see: Access-Control-Allow-Origin header
```

---

## Audit Logging Verification Checklist

### Log Content Checks
- [ ] All sensitive endpoints logged
- [ ] Client IP captured
- [ ] Timestamp in ISO 8601 format
- [ ] HTTP method and status code logged
- [ ] Response time in milliseconds
- [ ] Rate limit violations logged
- [ ] File uploads logged with hash

### Log Format Check
```bash
# Check recent logs (should be valid JSON)
tail -20 logs/app.json | jq '.' > /dev/null
# Should exit with status 0 (valid JSON)

# Verify password redaction
grep -i "password\|token\|secret" logs/app.json
# Should return nothing (no secrets in logs)
```

---

## Security Headers Verification Checklist

### Required Headers (Check with curl)
- [ ] X-Content-Type-Options: nosniff
- [ ] X-Frame-Options: DENY
- [ ] Content-Security-Policy: present
- [ ] Permissions-Policy: present
- [ ] Pragma: no-cache
- [ ] Expires: past date

### Verification Command
```bash
curl -I http://localhost:8504/query | grep -E "X-Content-Type|X-Frame|CSP|Permissions-Policy"

# Should show:
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# Content-Security-Policy: ...
# Permissions-Policy: ...
```

---

## Incident Response Checklist

### Immediate Actions (0-1 hour)
- [ ] Assess severity and scope
- [ ] Notify security team
- [ ] Enable enhanced logging
- [ ] Preserve evidence
- [ ] Initial containment if needed

### Investigation (1-4 hours)
- [ ] Review audit logs
- [ ] Identify attack vector
- [ ] Determine affected systems
- [ ] Collect forensic evidence
- [ ] Timeline reconstruction

### Remediation (4-24 hours)
- [ ] Apply security patch
- [ ] Update security rules
- [ ] Rotate affected credentials
- [ ] Deploy to all environments
- [ ] Verify fix effectiveness

### Post-Incident (1-7 days)
- [ ] Root cause analysis
- [ ] Preventive measures
- [ ] Update runbooks
- [ ] Team debriefing
- [ ] Document lessons learned

---

## Annual Security Requirements Checklist

### Penetration Testing
- [ ] Schedule external pen test
- [ ] Provide scope documentation
- [ ] Review findings
- [ ] Remediate critical/high findings
- [ ] Verify fixes

### Compliance Certification
- [ ] SOC2 Type II audit
- [ ] ISO27001 certification
- [ ] GDPR compliance verification
- [ ] Update documentation

### Team Training
- [ ] Security awareness training for all staff
- [ ] Secure coding training for developers
- [ ] Incident response drill
- [ ] Social engineering awareness

### Policy Review
- [ ] Update security policies
- [ ] Review incident response procedures
- [ ] Update vendor security requirements
- [ ] Update data retention policies

---

## Quick Troubleshooting

### Rate Limit Not Working?
1. Check /health endpoint (should return 1000 req/min)
2. Verify rate limiting middleware is enabled
3. Check Redis connectivity (if using Redis backend)
4. Review logs for rate limit bypass attempts

### Secrets Not Loading?
1. Verify 1Password/AWS Secrets Manager connectivity
2. Check IAM permissions (production AWS)
3. Verify secret names match config
4. Check logs for secret fetch errors

### Docker Security Options Not Applied?
1. Verify docker-compose.yml has security_opt
2. Rebuild image: `docker compose build --no-cache`
3. Remove old containers: `docker compose down`
4. Start fresh: `docker compose up -d`
5. Verify with: `docker inspect <container> | grep -A10 Security`

### Audit Logging Not Working?
1. Check logs directory exists and is writable
2. Verify logging configuration in config.py
3. Restart application to reload logging config
4. Check /var/log or application logs for errors
5. Verify JSON format with: `tail -1 logs/app.json | jq '.'`

---

## References

- **SECURITY.md** — Main security policy document
- **docs/security/AUDIT.md** — Complete audit findings
- **docs/security/REMEDIATION_PLAN.md** — Implementation details
- **docs/security/CVE_STATUS.md** — CVE tracking (if applicable)

---

**Last Updated:** 2026-03-16
**Review Frequency:** Quarterly
**Owner:** Security Team
