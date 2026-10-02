# Architecture Overview

This page explains how Nimbus Flow is built: the main services, where data lives, and what happens when a customer clicks a button. You don't need to memorize it. Come back to it whenever you touch a new part of the system.

## System at a glance

```
Browser
  │
  ▼
flow-web  (React + TypeScript single-page app)
  │  HTTPS / JSON
  ▼
flow-api  (Python 3.12, FastAPI)
  │            │
  ▼            ▼
PostgreSQL 16   Redis ──► flow-runner workers ──► connectors service ──► third-party apps
(primary DB)   (cache + job queue)
```

- **flow-web** is the React + TypeScript single-page app that customers use to build and monitor their flows. It is owned by Growth.
- **flow-api** is the Python 3.12 **FastAPI** backend. Everything the web app does goes through its REST API. It is owned by Platform.
- **PostgreSQL 16** is our **primary database**. It stores tenants, users, flows, run history and billing data.
- **Redis** is used for two things: a **cache** for hot data and the **job queue** for flow runs.
- **flow-runner** workers pick flow-run jobs off the Redis queue and execute each step of a customer's flow. They are owned by Platform.
- The **connectors service** knows how to talk to third-party apps (Google Sheets, Shopify, WhatsApp, …). It is owned by Integrations.

## Hosting

Everything runs on **AWS**:

- **ECS Fargate** runs our containers (flow-api, flow-runner, connectors). There are no servers to patch.
- **RDS** hosts our managed PostgreSQL database, with automatic backups.
- **S3** stores files, such as CSV uploads and exported reports.
- **CloudFront** serves the flow-web static files from locations close to our users.

Infrastructure is defined as code in the `nimbus/infra` repository. Changes there need two approvals (see *Engineering Workflow*).

## Request flow

What happens when a customer clicks **"Run now"** on a flow:

1. **flow-web** sends `POST /api/flows/{id}/runs` to flow-api with the user's session token.
2. **flow-api** checks the token, loads the flow from **PostgreSQL**, and confirms it belongs to the user's tenant.
3. flow-api creates a `flow_run` row with status `queued`, and pushes a job onto the **Redis** queue.
4. flow-api immediately returns `202 Accepted` with the run id. The web app shows a "running" spinner.
5. A **flow-runner** worker takes the job and executes each step. Steps that touch external apps call the **connectors service**.
6. The runner updates the `flow_run` row after each step (`running` → `succeeded` or `failed`).
7. flow-web polls the run status and shows the result to the customer.

## Multi-tenancy

Nimbus Flow is **multi-tenant**: many customer companies (tenants) share the same database.

- **Every table has a `tenant_id` column.**
- **Every query must filter by `tenant_id`.** Forgetting it could show one customer another customer's data. That would be a serious security incident.
- Our database helpers add the filter automatically when you use the `TenantSession`. Avoid writing raw SQL that bypasses it, and point it out in code review if you see it.

## Observability

How we know whether things are working:

- **Datadog**: metrics, dashboards and logs from every service.
- **Sentry**: errors and exceptions, with stack traces, grouped by issue.
- **PagerDuty**: alerts that wake up the on-call engineer when something important breaks.

Ask your onboarding buddy to show you the main Datadog dashboard in your first week. Watching real traffic is the fastest way to understand the system.
