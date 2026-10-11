# Approval interleaving audit

Status: prospective software audit design. The original approval-transfer reproduction is known. The replacement queue has not yet been inspected or tested by this audit author. This is an engineering check, not an RL experiment or a population safety estimate.

The revised contract assigns a unique handle and human token to each invocation. Concurrent invocations, including the same method and identical arguments, remain independently approvable. A token may release only its owning invocation, once. Unsupported argument values, changed arguments or policy, failed notification, cancellation and expiry must not grant authority. Notifications still show a summary; binding an invocation does not establish that a human reviewed every argument.

## Bounded order exploration

Use the actual exported queue with two distinct request handles, A and B, for the same session and method. Give them different immutable binding values. For each handle, include these four events exactly once:

1. Create the request.
2. Attach the first waiter.
3. Attach a duplicate waiter.
4. Apply the assigned terminal decision: approve, deny, or cancel.

Enumerate every distinct ordering of the eight labeled events in which creation precedes its other three events and the first waiter precedes its duplicate. Run all nine pairs of assigned terminal decisions. Derive the sequence count from these ordering constraints in the generator and independently verify uniqueness and coverage before execution. Do not sample or remove an ordering because of its observed result.

The first waiter is the only eligible consumer. It returns true if and only if its own request receives approval before being closed or consumed. The duplicate waiter always returns false. A request denied or cancelled before a waiter attaches remains unusable. Decisions for A cannot settle B. Both first waiters must succeed when each request receives its own approval. Token replay after settlement and explicit cleanup must return false for each request. A handle from another queue instance must not acquire or cancel an entry in this queue.

The test driver records promise settlement without blocking on a pending waiter. It flushes the JavaScript microtask queue after every event and compares observed settlements with a separately written state model. At the end, it cleans up both exact handles, awaits the remaining outcomes and checks no approval remains redeemable. Timers must be cancelled by cleanup so independent traces cannot affect each other. Every failing ordering is retained with its event list and expected/actual outcomes.

The implementation gate is fail-fast: the first counterexample stops that execution and is retained for repair. Only an execution completing all 5,670 allocated traces can pass. A stopped execution is not an estimate of a failure rate. The independently checked allocation count is 70 ways to interleave the two four-event sequences, three legal internal orders per request, and nine decision pairs: 70 × 3 × 3 × 9 = 5,670. A fresh execution identity is required after any driver or runtime repair.

This allocation explores two requests and these event types only. It does not prove arbitrary concurrency, external execution correctness, race freedom across processes, or behavior under every event-loop schedule. Expiry and delayed notification callbacks are separate integration fixtures; do not simulate those by merely renaming cancellation. Do not describe this as exhaustive verification of the entire runtime.

## Integration checks remain separate

The queue-order audit complements, but cannot replace, actual Interceptor and plugin fixtures for delayed callbacks, timeouts, mutable nested arguments, changed current policy, callback failures, authentic token routing and legitimate independent approvals. Those fixtures must invoke the guarded runtime and count only synthetic effects. No real notification, file mutation or external request is required.

The learning stop from the source-verification experiment remains unchanged. No model calls or reserved corpus access are part of this audit. Freeze the concrete test source, queue source and runtime identities before the first full order exploration; if a defect requires a repair, retain the failed artifact and give the subsequent audit a new execution identity.
