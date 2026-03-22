# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| Latest (`main`) | ✅ |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public GitHub issue.**

Email directly: [Wrishg@gmail.com](mailto:Wrishg@gmail.com)

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

You'll receive a response within 48 hours. If confirmed, a fix will be prioritised and you'll be credited in the release notes.

## Known Security Considerations

- **Session secret key** — must be set via `SECRET_KEY` environment variable in production. The fallback default in `app.py` is for local development only.
- **MongoDB URI** — never commit to version control. Use `.env` (see `.env.example`).
- **Uploaded files** — stored in `static/uploads/`, which is excluded from git via `.gitignore`.
