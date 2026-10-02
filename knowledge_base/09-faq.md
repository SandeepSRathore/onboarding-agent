# FAQ

Short answers to the questions new engineers ask most. Each answer points to the handbook page with the full details.

## Setup and access

**How do I get access to GitHub?**
Post your GitHub username in `#it-help` and ask to join the `nimbus` organization. It's usually approved within one working day. Details: *Tools & Access*.

**My VPN won't connect. What should I do?**
Quit and restart Tailscale, and make sure you signed in with Okta. If it still fails, post a screenshot in `#it-help`. Details: *Tools & Access*.

**Where do I find the secrets for my `.env.local` file?**
In the 1Password vault "Engineering – Dev". Never paste secrets into Slack. Details: *Development Environment Setup*.

**`make dev` fails with a database error. Help!**
Run `make migrate` to apply the latest migrations, and check Docker Desktop is running. Details: *Development Environment Setup*.

**When do I get AWS access?**
Read-only access through Okta after your first week; production write access after on-call training. Details: *Tools & Access*.

## Working here

**When is payday?**
The last working day of each month. Add your bank details in PeopleHub during week 1. Details: *HR Policies*.

**How do I apply for leave?**
In PeopleHub, at least one week in advance for planned leave. Details: *HR Policies*.

**Which days are office days?**
Tuesday to Thursday. Core hours are 11:00–16:00 IST. Details: *HR Policies*.

**Who should I ask when I'm stuck?**
Try for about 30 minutes, then ask your team channel or your onboarding buddy, Arjun Mehta. Details: *Your First 90 Days*.

**What happens at the 30-day check-in?**
A friendly conversation with your manager about what you've learned, what's been hard, and what support you need. Details: *Your First 90 Days*.

## Engineering

**How many approvals does my PR need?**
One approval for most repos, two for `infra`, plus green CI. Details: *Engineering Workflow*.

**When does my merged code reach production?**
Merges deploy to staging automatically. Production ships with the release train on Tuesdays and Thursdays at 14:00 IST. Details: *Engineering Workflow*.

**How should I name my branch?**
`feat/FLOW-123-short-description`, with `fix/` or `chore/` prefixes where they fit. Details: *Engineering Workflow*.

**What should my first PR be?**
A Jira ticket labelled `good-first-issue`, picked together with your buddy. Details: *Engineering Workflow*.

**I think I broke something in production. What do I do?**
Tell people immediately in `#incidents`. The on-call engineer will help. We have a blameless culture: we fix the system, not blame the person, and speaking up quickly is always the right call. Details: *Architecture Overview* (observability) and *Glossary* (on-call).

**I accidentally committed a secret. What now?**
Tell the Security team in `#security` right away so the key can be rotated. Details: *Tools & Access*.
