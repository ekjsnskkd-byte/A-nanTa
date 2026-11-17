# Security

## Threat model

Market feed poisoning, data leakage across time, malicious checkpoints, poisoned upstream dependencies, leaked API keys, unsafe automatic trading, missing human kill switch and exaggerated benchmark claims are first-class project risks.

## Controls

- Never commit broker keys, API tokens, passwords, private license emails or raw client data.
- Never install unverified model repos with arbitrary remote code as part of CI.
- Prefer pinned releases and verified checkpoint checksums; scan dependencies before deploying.
- Keep training/inference separate from real order placement.
- Treat model outputs as untrusted: validate NaNs, symbols, timestamps, units, prices and age.
- Keep live trading disabled until a separate reviewed execution service and risk gates exist.
- Use GitHub Secrets only if a future workflow truly requires credentials; CI should currently require none.

For sensitive disclosures, contact the repository owner privately; do not open a public issue containing secrets.
