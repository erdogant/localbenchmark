# Release process

Pulse is released every two weeks, on Tuesdays. Each release is announced to customers at least one day before it goes live.

Hotfixes can be released on any working day after approval from the on-call lead.

Every code change needs two approvals from other engineers before it can be merged. Changes that touch billing must also be behind a feature flag.

New code is deployed to the staging environment at least 24 hours before it reaches production.

Rollbacks are decided by the release manager. A rollback must be completed within 30 minutes of the decision.
