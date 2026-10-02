# Glossary

Words you'll hear a lot at Nimbus Labs, explained simply. If you hear a term that isn't here, ask in your team channel, then add it to this page!

## Company terms

**Flow**
: A customer's automation in Nimbus Flow, made of a *trigger* ("when a new order arrives…") and one or more *steps* ("…add a row to Google Sheets, then send an email"). Flows are the core thing customers build.

**Flow run**
: One execution of a flow. Every time the trigger fires, a new flow run is created, and its status moves from `queued` to `running` to `succeeded` or `failed`.

**Runner**
: Short for the `flow-runner` workers: background processes that take flow-run jobs from the queue and execute each step.

**Tenant**
: One customer company and all of its data. Many tenants share our database, so every table has a `tenant_id` column, and every query must filter by it.

**Connector**
: The code that lets a flow talk to a third-party app such as Google Sheets, Shopify or WhatsApp. Connectors are owned by the Integrations team.

## Engineering terms

**PR (pull request)**
: A request to merge your branch into `main`. Teammates review the code in the PR before it's merged.

**CI/CD (continuous integration / continuous delivery)**
: Automation that tests every change (CI, using GitHub Actions) and deploys merged changes (CD, to staging automatically and to production via the release train).

**Squash merge**
: Merging a PR so that all its commits are combined into one commit on `main`. It keeps history clean.

**Staging**
: A production-like environment. Every merge to `main` deploys there automatically, so we can check changes before customers see them.

**Feature flag**
: A switch that turns a feature on or off without deploying new code. We use flags to release features gradually, or to hide unfinished work that's already merged.

**SLO (service level objective)**
: A reliability target we commit to. For example, "99.9% of flow runs start within 60 seconds". If we're at risk of missing an SLO, reliability work takes priority over new features.

## Process terms

**Release train**
: Our scheduled **production deploy on Tuesdays and Thursdays at 14:00 IST**. Everything merged and verified on staging "boards the train" and ships together. If you miss one, the next train is at most a few days away.

**On-call**
: The engineer responsible for responding to production alerts for a week. The rotation changes every week. New engineers shadow on-call before joining the rotation.

**Standup**
: The daily 15-minute team meeting (10:30 IST) where everyone shares progress, plans and blockers.

**Sprint**
: A two-week block of work. The team plans which tickets to finish at sprint planning, and reflects at the retro.

**Retro (retrospective)**
: A meeting every other Friday where the team discusses what went well, what didn't, and what to change.

**Good first issue**
: A Jira ticket labelled `good-first-issue`: small, well-defined and safe. A great choice for your first PR.
