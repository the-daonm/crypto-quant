# Security policy

## Supported versions

Only the latest minor release receives fixes.

| Version | Supported |
|---|---|
| 0.2.x | ✅ |
| < 0.2 | ❌ |

## Reporting a vulnerability

Please **do not open a public issue** for security problems.
Report them privately via GitHub's
[private vulnerability reporting](https://github.com/the-daonm/crypto-quant/security/advisories/new).

Include a description of the issue, steps to reproduce, and the affected version or commit.
You can expect an initial response within 7 days. Once a fix is available, we will publish an advisory
and credit you unless you prefer to remain anonymous.

## Scope

crypto-quant only reads **public** market data and needs no API keys, accounts or personal data.
Relevant reports include, for example, unsafe handling of downloaded data, dependency vulnerabilities,
or CI workflows that could leak tokens. Trading losses from using the strategy are not security issues:
this is an educational project and not investment advice.
