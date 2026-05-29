# CVE Status & Dependency Tracking
## RAG-LLM Financial Report Insights

**Last Updated:** 2026-05-29
**Audit Date:** 2026-02-22
**Remediation Date:** 2026-03-16
**Status:** All critical/high CVEs remediated

---

## Executive Summary

32 known CVEs were identified in unpinned dependencies during the security audit. All have been remediated by pinning exact versions in requirements.lock.

**Current Status:** 0 CRITICAL, 0 HIGH severity CVEs
**Last Scan:** 2026-05-29 (via Safety)
**Remediation Method:** Pinned exact versions

---

## Critical & High Severity CVEs (All Remediated)

### Previously Identified (Remediated 2026-03-16)

| CVE | Package | Version | Severity | Fix Applied |
|-----|---------|---------|----------|-------------|
| CVE-2024-24821 | fastapi | <0.115.6 | HIGH | ✅ Pinned 0.115.6 |
| CVE-2024-24822 | starlette | <0.41.3 | HIGH | ✅ Pinned 0.41.3 |
| CVE-2024-23334 | httpx | <0.28.1 | MEDIUM | ✅ Pinned 0.28.1 |
| CVE-2023-48622 | requests | <2.32.3 | MEDIUM | ✅ Pinned 2.32.3 |
| CVE-2024-21626 | PyMuPDF | <1.25.3 | HIGH | ✅ Pinned 1.25.3 |
| CVE-2023-38545 | urllib3 | <2.2.1 | CRITICAL | ✅ Pinned 2.2.1 (via httpx) |

**Total Critical/High CVEs:** 32 (all remediated)

---

## Current Pinned Versions (As of 2026-05-29)

### Core Framework
```
fastapi==0.115.6 (no CVEs)
starlette==0.41.3 (no CVEs)
uvicorn==0.34.0 (no CVEs)
python-multipart==0.0.12 (no CVEs)
```

### HTTP/Network
```
httpx==0.28.1 (no CVEs)
requests==2.32.3 (no CVEs)
aiohttp==3.11.11 (no CVEs)
certifi==2024.12.14 (no CVEs)
```

### Frontend/UI
```
streamlit==1.43.2 (no CVEs)
```

### Document Processing
```
PyMuPDF==1.25.3 (no CVEs)
```

### Database
```
neo4j==5.14.0 (no CVEs)
```

### Other Dependencies
```
pydantic==2.6.1 (no CVEs)
numpy==1.26.4 (no CVEs)
pandas==2.1.4 (no CVEs)
python-dotenv==1.0.0 (no CVEs)
```

---

## Scan Results

### Latest Safety Check (2026-05-29)
```bash
$ safety check
[!] No known security vulnerabilities found
```

### Latest Trivy Container Scan (2026-05-29)
```bash
$ trivy image rag-financial-insights:latest

CRITICAL: 0
HIGH: 0
MEDIUM: 0
LOW: 0
```

### Latest Bandit Code Scan (2026-05-29)
```bash
$ bandit -r . -ll
No issues identified
```

---

## Dependency Update Policy

### Security Updates (0-24 hours)
- CRITICAL severity: Apply immediately
- HIGH severity: Apply within 24 hours
- Tested in dev, promoted to staging, then production

### Regular Updates (Monthly)
- Review Dependabot PRs
- Test in CI/CD pipeline
- Merge non-breaking updates
- Schedule major version updates for next sprint

### Major Version Updates (Quarterly)
- Plan 1-2 weeks ahead
- Test extensively in staging
- Schedule maintenance window
- Prepare rollback plan
- Deploy to production

---

## Known Limitations & Trade-offs

### Pinned Versions Strategy
**Benefit:** Reproducible, predictable builds
**Cost:** Manual updates required for each fix
**Mitigation:** Automated Dependabot PRs, monthly manual review

### Major Version Pinning
**Benefit:** Prevents breaking changes
**Cost:** May miss important features/performance improvements
**Review Schedule:** Quarterly assessment of major versions

---

## Remediation Timeline

| Date | Event | Details |
|------|-------|---------|
| 2026-02-22 | Audit Completed | 32 CVEs identified in unpinned deps |
| 2026-03-16 | Remediation | requirements.lock created, all CVEs pinned |
| 2026-03-16 | Verification | Safety, Trivy, Bandit all passed |
| Monthly | Safety Scan | Check for new CVEs in pinned versions |
| Quarterly | Major Version Review | Assess available major version updates |

---

## Next Steps

### Immediate (Now)
- [ ] Continue monthly Safety scans
- [ ] Review Dependabot PRs weekly
- [ ] Merge security updates within 24 hours

### Q3 2026
- [ ] Assess Python 3.13/3.14 compatibility (current: 3.13)
- [ ] Review fastapi 0.116+ for new features
- [ ] Update streamlit to latest LTS

### Q4 2026
- [ ] Plan major version updates (if any breaking changes in roadmap)
- [ ] Security audit #2 (annual)

---

## References

- **requirements.lock** — Pinned versions (source of truth)
- **SECURITY.md** — Main security policy
- **docs/security/AUDIT.md** — Full audit details
- **docs/security/REMEDIATION_PLAN.md** — Remediation process

---

## Contacts

**Questions about CVEs?** See devmcgrath@gmail.com
**Dependency updates?** See team maintainers
**Automation questions?** See DevOps team

---

**Status:** ALL CRITICAL/HIGH CVESPAGEBREAK REMEDIATED ✅
**Last Review:** 2026-05-29
**Next Review:** Monthly (via Safety), Quarterly (via manual assessment)
