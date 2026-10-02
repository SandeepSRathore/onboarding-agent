# Development Environment Setup

This guide takes you from a fresh laptop to running Nimbus Flow on your machine. Expect it to take most of an afternoon. If you get stuck on any step for more than 30 minutes, ask your onboarding buddy.

Before you start, make sure you have GitHub access and the Tailscale VPN working. Both are explained in *Tools & Access*. You can't clone our private repositories without them.

## Prerequisites

Your MacBook comes from IT with macOS and admin rights. Install these tools in order.

**Homebrew**, the macOS package manager:

```bash
# install Homebrew (follow the instructions it prints at the end)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Git, uv and fnm:**

```bash
# git for version control, uv for Python, fnm for Node.js versions
brew install git uv fnm
```

**Python 3.12** (managed by `uv`) and **Node 22** (managed by `fnm`):

```bash
# install the exact language versions our repos use
uv python install 3.12
fnm install 22 && fnm default 22
```

**Docker Desktop:** download it from the Docker website, install it and start it once. We use Docker to run Postgres and Redis locally.

## Clone the repositories

We have three main repositories in the `nimbus` GitHub organization:

- `nimbus/flow-api`: the Python backend
- `nimbus/flow-web`: the React frontend
- `nimbus/infra`: infrastructure as code (you won't need this in week 1)

```bash
# keep all Nimbus code in one folder
mkdir -p ~/code && cd ~/code
git clone git@github.com:nimbus/flow-api.git
git clone git@github.com:nimbus/flow-web.git
```

If cloning fails with "permission denied", check two things: you are connected to the VPN, and your GitHub access request was approved (see *Tools & Access*).

## Secrets and environment files

Each repo has a `.env.example` file that lists the settings it needs. Copy it to `.env.local` and fill in the values from the **1Password vault "Engineering – Dev"**:

```bash
cd ~/code/flow-api
cp .env.example .env.local
# now open .env.local and paste the values from 1Password
```

Never commit `.env.local` (it is already in `.gitignore`), and never paste secrets into Slack.

## Run flow-api

`flow-api` needs Postgres and Redis. Docker Compose starts both:

```bash
cd ~/code/flow-api
# start Postgres and Redis in the background
docker compose up -d
# install dependencies, create the database and load sample data
make bootstrap
# start the API with auto-reload
make dev
```

The API is now running at **http://localhost:8080**. Open http://localhost:8080/docs to see the interactive API documentation.

## Run flow-web

In a second terminal:

```bash
cd ~/code/flow-web
# install exact dependency versions from the lockfile
npm ci
# start the dev server
npm run dev
```

Open **http://localhost:5173**. Log in with the sample account `demo@nimbuslabs.example`. The password is in the same 1Password vault.

## Troubleshooting

**"Port 8080 is already in use."** Something else is running on that port. Find it with `lsof -i :8080` and stop it, or stop an old `make dev` you forgot about.

**"Cannot connect to the Docker daemon."** Docker Desktop isn't running. Open it from Applications and wait until the whale icon stops animating.

**Database errors after pulling new code.** Someone added a database migration. Run:

```bash
# apply the latest database migrations
make migrate
```

**`npm ci` fails.** Check `node --version` shows v22. If not, run `fnm use 22`.

Still stuck? Post the exact error message in your team channel, or ask your onboarding buddy. Screenshots of the error help a lot.
