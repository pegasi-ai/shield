import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const scratch = mkdtempSync(join(tmpdir(), 'reins-approval-binding-'));
process.env.OPENCLAW_HOME = scratch;
Object.defineProperty(process.stdin, 'isTTY', { value: false, configurable: true });
const require = createRequire(import.meta.url);
const { bindApprovalArguments, APPROVAL_MAX_BYTES } = require('../dist/core/ApprovalBinding.js');
const { ApprovalQueue } = require('../dist/core/ApprovalQueue.js');
const { Arbitrator } = require('../dist/core/Arbitrator.js');
const { Interceptor } = require('../dist/core/Interceptor.js');
const { DecisionLog } = require('../dist/storage/DecisionLog.js');
const { logger } = require('../dist/core/Logger.js');
logger.silent = true;
test.after(() => {
  logger.close();
  rmSync(scratch, { recursive: true, force: true });
});
const request = (queue, binding = 'checked') =>
  queue.request('session', 'FileSystem', 'write', {
    binding,
    actionSummary: 'Review this invocation',
  });
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fixture(ttl = 1000) {
  const queue = new ApprovalQueue(ttl);
  const policy = {
    defaultAction: 'DENY',
    modules: { FileSystem: { write: { action: 'ASK', allowPaths: ['/scratch/**'] } } },
  };
  const interceptor = new Interceptor(policy, false);
  interceptor.arbitrator = new Arbitrator(queue);
  return { queue, policy, interceptor };
}
function start(interceptor, args, intervention) {
  return interceptor.evaluate('FileSystem', 'write', args, 'same-session', intervention).then(
    () => ({ allowed: true }),
    (error) => ({ allowed: false, error: error.message })
  );
}

test('binding sorts object keys, preserves exact values, and detaches/freezes the snapshot', () => {
  const original = [{ z: ['é', -0, 1], a: { yes: true } }];
  const bound = bindApprovalArguments(original);
  assert.equal(
    bound.canonical,
    bindApprovalArguments([{ a: { yes: true }, z: ['é', -0, 1] }]).canonical
  );
  assert.notEqual(
    bound.canonical,
    bindApprovalArguments([{ a: { yes: true }, z: ['é', 0, 1] }]).canonical
  );
  assert(Object.is(bound.snapshot[0].z[1], -0));
  original[0].a.yes = false;
  assert.equal(bound.snapshot[0].a.yes, true);
  assert(Object.isFrozen(bound.snapshot[0].a));
  assert.throws(() => {
    bound.snapshot[0].z.push('changed');
  }, TypeError);
});

test('unsupported values never execute getters, toJSON, or Proxy traps', async (t) => {
  let sideEffects = 0;
  const cycle = {};
  cycle.self = cycle;
  const accessor = Object.defineProperty({}, 'path', {
    enumerable: true,
    get() {
      sideEffects += 1;
      return '/scratch/a';
    },
  });
  const proxy = new Proxy(
    {},
    {
      ownKeys() {
        sideEffects += 1;
        return [];
      },
      getPrototypeOf() {
        sideEffects += 1;
        return Object.prototype;
      },
      get() {
        sideEffects += 1;
      },
    }
  );
  const arrayExtra = [];
  arrayExtra.extra = 1;
  const hidden = Object.defineProperty({}, 'hidden', { value: 1 });
  const cases = [
    undefined,
    () => 1,
    Symbol('a'),
    1n,
    NaN,
    Infinity,
    cycle,
    accessor,
    proxy,
    {
      toJSON() {
        sideEffects += 1;
        return {};
      },
    },
    new Date(),
    [, 1],
    arrayExtra,
    hidden,
    { [Symbol('key')]: 1 },
  ];
  for (const [i, value] of cases.entries())
    await t.test(String(i), () => assert.throws(() => bindApprovalArguments([value])));
  assert.equal(sideEffects, 0);
});

test('binding enforces documented depth and UTF-8 size bounds', () => {
  let value = 1;
  for (let i = 0; i < 65; i += 1) value = [value];
  assert.throws(() => bindApprovalArguments(value), /depth/);
  assert.throws(() => bindApprovalArguments('x'.repeat(APPROVAL_MAX_BYTES)), /1 MiB/);
  assert.throws(() => bindApprovalArguments('é'.repeat(APPROVAL_MAX_BYTES / 2)), /1 MiB/);
  assert.equal(
    bindApprovalArguments('x'.repeat(APPROVAL_MAX_BYTES - 2)).canonical.length,
    APPROVAL_MAX_BYTES
  );
});

test('each identical request has a separate token, owner, and one-use waiter', async () => {
  const queue = new ApprovalQueue(1000);
  const a = request(queue);
  const b = request(queue);
  const at = queue.getNotificationInfo(a.handle).token;
  const bt = queue.getNotificationInfo(b.handle).token;
  assert.notEqual(at, bt);
  assert(Object.isFrozen(a) && Object.isFrozen(a.handle));
  const wa = queue.waitForApproval(a.handle);
  const wb = queue.waitForApproval(b.handle);
  assert.equal(await queue.waitForApproval(a.handle), false);
  assert.equal(queue.getNotificationInfo({ id: a.handle.id }), undefined);
  assert.equal(queue.cancel({ id: b.handle.id }), false);
  assert.equal(queue.resolveByToken(at, 'approve'), true);
  assert.equal(await wa, true);
  assert.equal(queue.resolveByToken(at, 'approve'), false);
  assert.equal(queue.getNotificationInfo(b.handle).token, bt);
  assert.equal(queue.resolveByToken(bt, 'deny'), true);
  assert.equal(await wb, false);
  assert.equal(await queue.waitForApproval(a.handle), false);
});

test('unbound legacy request is rejected and supplied tokens cannot choose a channel token', () => {
  const queue = new ApprovalQueue(1000);
  assert.equal(queue.request('s', 'm', 'f', { confirmationToken: 'CONFIRM-OLD' }), undefined);
  const a = queue.request('s', 'm', 'f', { binding: 'args', confirmationToken: 'CONFIRM-OLD' });
  assert.notEqual(queue.getNotificationInfo(a.handle).token, 'CONFIRM-OLD');
  assert.equal(queue.resolveByToken('CONFIRM-OLD', 'approve'), false);
  queue.cancel(a.handle);
});

test('approval does not extend expiry and never reaches a later invocation', async () => {
  const queue = new ApprovalQueue(25);
  const a = request(queue);
  const token = queue.getNotificationInfo(a.handle).token;
  assert.equal(queue.resolveByToken(token, 'approve'), true);
  await delay(40);
  assert.equal(await queue.waitForApproval(a.handle), false);
  const b = request(queue);
  assert.equal(queue.cancel(a.handle), false);
  assert(queue.getNotificationInfo(b.handle));
  assert.equal(queue.resolveByToken(token, 'approve'), false);
  queue.cancel(b.handle);
});

test('failed, throwing, and hanging callbacks cannot leave approvals usable', async (t) => {
  for (const kind of ['false', 'throw', 'hang'])
    await t.test(kind, async () => {
      const queue = new ApprovalQueue(20);
      const a = request(queue);
      const token = queue.getNotificationInfo(a.handle).token;
      const notified = await queue.notify(a.handle, async () => {
        queue.resolveByToken(token, 'approve');
        if (kind === 'throw') throw new Error('send failed');
        if (kind === 'hang') return new Promise(() => {});
        return false;
      });
      assert.equal(notified, false);
      assert.equal(await queue.waitForApproval(a.handle), false);
      assert.equal(queue.resolveByToken(token, 'approve'), false);
    });
});

test('changed-argument call cannot consume A approval, while both can receive independent approvals', async () => {
  const { queue, interceptor } = fixture();
  const seen = [];
  const ready = deferred();
  interceptor.onBlockCallback = async (_s, _m, _f, context) => {
    seen.push(context);
    if (seen.length === 2) ready.resolve();
    return true;
  };
  const a = start(interceptor, [{ path: '/scratch/a', content: 'reviewed' }]);
  const b = start(interceptor, [{ path: '/scratch/b', content: 'different' }]);
  await ready.promise;
  assert.notEqual(seen[0].binding, seen[1].binding);
  const at = queue.getNotificationInfo(seen[0].handle).token;
  const bt = queue.getNotificationInfo(seen[1].handle).token;
  queue.resolveByToken(at, 'approve');
  assert.deepEqual(await a, { allowed: true });
  assert.equal(queue.getNotificationInfo(seen[1].handle).token, bt);
  queue.resolveByToken(bt, 'approve');
  assert.deepEqual(await b, { allowed: true });
});

test('post-approval mutation of args or current policy is denied', async (t) => {
  for (const change of ['args', 'deny', 'ask-scope', 'intervention'])
    await t.test(change, async () => {
      const { queue, policy, interceptor } = fixture();
      const args = [{ path: '/scratch/a', nested: { text: 'checked' } }];
      const intervention = { actionSummary: 'One write', requiresExplicitConfirmation: true };
      interceptor.onBlockCallback = async (_s, _m, _f, context) => {
        const token = queue.getNotificationInfo(context.handle).token;
        if (change === 'args') args[0].nested.text = 'changed';
        if (change === 'deny') policy.modules.FileSystem.write.action = 'DENY';
        if (change === 'ask-scope') policy.modules.FileSystem.write.allowPaths.push('/another/**');
        if (change === 'intervention') intervention.requiresExplicitConfirmation = false;
        queue.resolveByToken(token, 'approve');
        return true;
      };
      assert.equal((await start(interceptor, args, intervention)).allowed, false);
    });
});

test('truthy malformed forceAsk still receives channel binding and final guards', async () => {
  const { queue, policy, interceptor } = fixture();
  policy.modules.FileSystem.write.action = 'ALLOW';
  const args = [{ path: '/scratch/a', content: 'initial' }];
  interceptor.onBlockCallback = async (_s, _m, _f, context) => {
    args[0].content = 'changed';
    queue.resolveByToken(queue.getNotificationInfo(context.handle).token, 'approve');
    return true;
  };
  assert.equal((await start(interceptor, args, { forceAsk: 'yes' })).allowed, false);
});

test('mutation during approval logging is rejected and logs retain the detached approved args', async () => {
  const { queue, interceptor } = fixture();
  const args = [{ path: '/scratch/original', content: 'checked' }];
  const records = [];
  const append = DecisionLog.append;
  DecisionLog.append = async (record) => {
    if (record.decision === 'APPROVED') args[0].content = 'changed during logging';
    records.push(JSON.parse(JSON.stringify(record)));
  };
  try {
    interceptor.onBlockCallback = async (_s, _m, _f, context) => {
      queue.resolveByToken(queue.getNotificationInfo(context.handle).token, 'approve');
      return true;
    };
    assert.equal((await start(interceptor, args)).allowed, false);
    assert.equal(records[0].decision, 'APPROVED');
    assert.equal(records[0].args[0].content, 'checked');
    assert.equal(records.at(-1).decision, 'REJECTED');
    assert.match(records.at(-1).reason, /arguments or current policy changed/);
  } finally {
    DecisionLog.append = append;
  }
});

test('unsupported ASK args fail before target getter and log a controlled rejection', async () => {
  const { interceptor } = fixture();
  let touches = 0;
  const args = [
    Object.defineProperty({}, 'path', {
      enumerable: true,
      get() {
        touches += 1;
        return '/scratch/a';
      },
    }),
  ];
  const result = await start(interceptor, args);
  assert.equal(result.allowed, false);
  assert.match(result.error, /invalid channel approval arguments/);
  assert.equal(touches, 0);
});

test('callback timeout bounds evaluation and stale callback cannot alter a newer call', async () => {
  const { queue, interceptor } = fixture(25);
  const delayed = deferred();
  let oldContext;
  interceptor.onBlockCallback = async (_s, _m, _f, context) => {
    oldContext = context;
    return delayed.promise;
  };
  const old = await start(interceptor, [{ path: '/scratch/a' }]);
  assert.equal(old.allowed, false);
  interceptor.onBlockCallback = async (_s, _m, _f, context) => {
    delayed.resolve(true);
    assert.equal(queue.cancel(oldContext.handle), false);
    queue.resolveByToken(queue.getNotificationInfo(context.handle).token, 'approve');
    return true;
  };
  assert.equal((await start(interceptor, [{ path: '/scratch/a' }])).allowed, true);
});
