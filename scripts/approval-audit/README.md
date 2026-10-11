# Check approval ownership across request orderings

From the repository root, after `npm ci --ignore-scripts` and `npm run build`:

```sh
node scripts/approval-audit/interleaving_audit.mjs self-test
node scripts/approval-audit/interleaving_audit.mjs freeze "$PWD" /tmp/reins-approval-audit
node scripts/approval-audit/interleaving_audit.mjs run "$PWD" /tmp/reins-approval-audit
```

Choose a new output directory for each run. The driver refuses to overwrite results and rejects changed source or runtime files. It records the Node binary, queue source, built JavaScript, installed JavaScript dependencies, protocol, and complete allocation before execution, then checks those identities again after execution.

The audit enumerates all 630 valid orderings of two requests' creation, first wait, duplicate wait, and decision. Crossing those orderings with approve, deny, and cancel for each request produces 5,670 traces. Each trace compares the actual queue with an independent state model. Approving one request must never release the other; approving both must release both. It also checks token replay, foreign handles, and cleanup.

This directory preserves the tested driver and original pre-execution protocol verbatim. The protocol's prospective wording records when the design was made. The original learning-study references are background and are not prerequisites for running this software check.

A complete passing run establishes this bounded two-request contract. It does not prove behavior under arbitrary concurrency, every event-loop schedule, or external side effects. The normal `npm test` suite separately covers notification failures, expiry, argument changes, policy changes, and the Interceptor integration. All effects used in these checks are synthetic; no approval message is sent to a real person.
