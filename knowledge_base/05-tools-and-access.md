# Tools & Access

This page lists every account and tool you need, how to get access, and when. Most problems here are solved by posting in **#it-help** on Slack.

## Day-one accounts

IT sets these up before you arrive. On your first day, sign in to each one:

- **Okta (single sign-on):** one login for almost every company tool. On first login, set up **multi-factor authentication (MFA) with the Okta Verify app** on your phone. Okta is the front door to everything else, so do this first.
- **Google Workspace:** your `@nimbuslabs.example` email, calendar and Google Drive.
- **Slack:** our main chat tool. Join `#general`, `#it-help`, your team channel (for example `#team-platform`), and `#engineering`.

If any of these don't work, ask IT in person during laptop pickup or post in `#it-help`.

## GitHub

Our code lives in the **`nimbus`** GitHub organization.

1. Create a personal GitHub account if you don't have one. Using your existing account is fine.
2. Turn on two-factor authentication in GitHub settings. It's required by the organization.
3. **Post your GitHub username in `#it-help`** and ask to be added to the `nimbus` organization and your team's GitHub team.
4. Access is usually **approved within 1 working day**. You'll get an email invitation; accept it.
5. Add an SSH key to GitHub so you can clone repositories (GitHub's "Generating a new SSH key" guide walks you through it).

## VPN

Internal tools and our private repositories are only reachable through the **Tailscale VPN**.

1. Install **Tailscale** from the Mac App Store or with `brew install --cask tailscale`.
2. Open it and **sign in with Okta** (choose "Sign in with OIDC/SSO" and use your work email).
3. Make sure the Tailscale menu-bar icon shows *Connected*.

You need the VPN connected to **clone private repos, open internal dashboards, and reach staging**. If the VPN won't connect, quit and restart Tailscale first. If that doesn't help, post in `#it-help` with a screenshot.

## Passwords and secrets

- We use **1Password** for all passwords and shared secrets. Your account is created on day one, and you'll be added to the vaults your team needs (for engineers, that includes "Engineering – Dev").
- **Never share secrets in Slack**, email, Jira or code. Share a 1Password item instead.
- If you accidentally paste or commit a secret, tell the Security team immediately in `#security`. Rotating a leaked key quickly matters more than embarrassment, and nobody will blame you.

## Other tools

You'll get access to these through Okta during week 1:

- **Jira:** tickets, sprints and bug reports. Your team board is where your work comes from.
- **Confluence:** design documents, meeting notes and team pages.
- **Datadog:** metrics, dashboards and logs.
- **Sentry:** application errors.

## AWS access

- **Week 1:** no AWS access needed; everything runs locally.
- **After week 1:** you get **read-only AWS console access through Okta** (open the AWS tile in your Okta dashboard). It's enough to look at logs, queues and dashboards.
- **Production write access** is granted only **after you complete on-call training**, usually around your third month.

## Laptop

Your MacBook is **issued by IT** and remains company property.

- Keep macOS updated. IT pushes security updates automatically, so restart when asked.
- Report hardware or software problems in `#it-help`. For an urgent problem (the laptop won't boot), walk over to the IT desk.
- Don't install unlicensed software, and lock your screen (Ctrl+Cmd+Q) when you walk away.
