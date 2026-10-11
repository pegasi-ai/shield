import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { ApprovalQueue } = require('../dist/core/ApprovalQueue.js');

test('a clock tick while shortening the wait cannot extend the original expiry', async () => {
  const originalNow = Date.now;
  const queue = new ApprovalQueue(100);
  let phase = 'request';
  let waitClockReads = 0;
  Date.now = () => {
    if (phase === 'request') return 1000;
    if (phase === 'wait') {
      waitClockReads += 1;
      return waitClockReads <= 2 ? 1000 : 1002;
    }
    return 1100; // Exact original expiry; approval must no longer be usable.
  };
  let request;
  let waited;
  let resolved;
  try {
    request = queue.request('deadline-session', 'Module', 'method', { binding: 'args' });
    const token = queue.getNotificationInfo(request.handle).token;
    phase = 'wait';
    waited = queue.waitForApproval(request.handle, 99);
    phase = 'original-expiry';
    resolved = queue.resolveByToken(token, 'approve');
  } finally {
    Date.now = originalNow;
    if (request) queue.cancel(request.handle);
  }
  assert.equal(resolved, false, 'shorter timeout must not make approval usable at original expiry');
  assert.equal(await waited, false);
});
