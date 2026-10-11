// Independently authored from the baseline and API/design contract before
// implementation inspection. Synthetic counters only; no tool effects.
import assert from 'node:assert/strict';
import { after, test } from 'node:test';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const scratch = mkdtempSync(join(tmpdir(), 'reins-independent-approval-'));
const priorHome = process.env.OPENCLAW_HOME;
const priorTTY = Object.getOwnPropertyDescriptor(process.stdin, 'isTTY');
process.env.OPENCLAW_HOME = scratch;
Object.defineProperty(process.stdin, 'isTTY', { value: false, configurable: true });
const require = createRequire(import.meta.url);
const { logger } = require('../dist/core/Logger.js');
const { ApprovalQueue } = require('../dist/core/ApprovalQueue.js');
const { Arbitrator } = require('../dist/core/Arbitrator.js');
const { Interceptor } = require('../dist/core/Interceptor.js');
const { DecisionLog } = require('../dist/storage/DecisionLog.js');
logger.silent = true;

after(() => {
  logger.close();
  if (priorHome === undefined) delete process.env.OPENCLAW_HOME;
  else process.env.OPENCLAW_HOME = priorHome;
  if (priorTTY) Object.defineProperty(process.stdin, 'isTTY', priorTTY);
  else delete process.stdin.isTTY;
  rmSync(scratch, { recursive: true, force: true });
});

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

const argsA = () => [{ path: '/synthetic/reviewed.txt', content: 'reviewed content',
  options: { encoding: 'utf8', append: false } }];
const argsB = () => [{ path: '/synthetic/other.txt', content: 'different content',
  options: { encoding: 'utf8', append: false } }];
const policyFor = () => ({ defaultAction: 'DENY', modules: {
  FileSystem: { write: { action: 'ASK', description: 'Review this invocation.' } },
} });
let serial = 0;

function harness(t, { ttl = 2000, policy = policyFor() } = {}) {
  const queue = new ApprovalQueue(ttl);
  const interceptor = new Interceptor(policy, false);
  // Agreed test seam: inject the public Arbitrator with a test-local queue.
  interceptor.arbitrator = new Arbitrator(queue);
  const notices = [];
  const arrivals = new Map();
  const gates = [];
  const session = `independent-approval-${++serial}`;
  const h = { queue, interceptor, notices, policy, session, onNotice: () => true };
  interceptor.onBlockCallback = async (sessionKey, moduleName, methodName, context) => {
    assert.ok(context?.handle, 'Exact request context is required');
    const info = queue.getNotificationInfo(context.handle);
    assert.equal(typeof info?.token, 'string');
    assert.ok(info.token.length > 0);
    const notice = { sessionKey, moduleName, methodName, context, ...info };
    const index = notices.push(notice) - 1;
    arrivals.get(index)?.resolve(notice);
    return h.onNotice(notice, index);
  };
  h.notice = (index) => {
    if (notices[index]) return Promise.resolve(notices[index]);
    if (!arrivals.has(index)) arrivals.set(index, deferred());
    return arrivals.get(index).promise;
  };
  h.gate = () => { const gate = deferred(); gates.push(gate); return gate; };
  h.start = (args = argsA(), options = {}) => {
    const operation = { state: 'pending', effects: 0 };
    operation.result = interceptor.evaluate(options.moduleName ?? 'FileSystem',
      options.methodName ?? 'write', args, options.session ?? session,
      options.intervention ?? { actionSummary: 'Synthetic request for review.' }).then(
      () => { operation.state = 'allowed'; operation.effects += 1; return { allowed: true }; },
      (error) => { operation.state = 'denied'; return { allowed: false, error: String(error.message) }; },
    );
    return operation;
  };
  h.decide = (notice, decision = 'approve') => {
    assert.equal(queue.resolveByToken(notice.token, decision), true);
  };
  t.after(() => {
    for (const notice of notices) queue.cancel(notice.context.handle);
    for (const gate of gates) gate.resolve(false);
  });
  return h;
}

function noDisclosure(result, notices) {
  assert.equal(result.allowed, false);
  for (const notice of notices) {
    assert.ok(!result.error.includes(notice.token), 'Approval token leaked in denial');
    if (typeof notice.context.handle.id === 'string') {
      assert.ok(!result.error.includes(notice.context.handle.id), 'Owner handle leaked in denial');
    }
  }
}

test('independent: approved A cannot transfer authority to concurrent changed B', { timeout: 4000 }, async (t) => {
  const h = harness(t);
  const gate = h.gate();
  h.onNotice = (_, index) => index === 0 ? gate.promise : true;
  const a = h.start();
  const na = await h.notice(0);
  h.decide(na);
  const b = h.start(argsB());
  const nb = await h.notice(1);
  assert.notEqual(na.token, nb.token);
  assert.notEqual(na.context.handle, nb.context.handle);
  gate.resolve(true);
  assert.equal((await a.result).allowed, true);
  assert.equal(b.state, 'pending');
  assert.deepEqual([a.effects, b.effects], [1, 0]);
  h.decide(nb);
  assert.equal((await b.result).allowed, true);
  assert.deepEqual([a.effects, b.effects], [1, 1]);
});

for (const identical of [false, true]) {
  test(`independent: pending ${identical ? 'identical' : 'different'} requests need separate decisions`, { timeout: 4000 }, async (t) => {
    const h = harness(t);
    const a = h.start();
    const na = await h.notice(0);
    const b = h.start(identical ? argsA() : argsB());
    const nb = await h.notice(1);
    assert.notEqual(na.token, nb.token);
    h.decide(na);
    assert.equal((await a.result).allowed, true);
    assert.equal(b.state, 'pending');
    h.decide(nb, 'deny');
    noDisclosure(await b.result, h.notices);
    assert.deepEqual([a.effects, b.effects], [1, 0]);
  });
}

test('independent: unchanged request and later identical retry each receive a fresh approval', { timeout: 4000 }, async (t) => {
  const h = harness(t);
  const a = h.start(); const na = await h.notice(0); h.decide(na);
  assert.equal((await a.result).allowed, true);
  const b = h.start(); const nb = await h.notice(1);
  assert.notEqual(na.token, nb.token);
  assert.equal(h.queue.resolveByToken(na.token, 'approve'), false);
  assert.equal(b.state, 'pending');
  h.decide(nb);
  assert.equal((await b.result).allowed, true);
  assert.deepEqual([a.effects, b.effects], [1, 1]);
});

test('independent: two sessions remain separately approvable', { timeout: 4000 }, async (t) => {
  const h = harness(t);
  const a = h.start(argsA(), { session: `${h.session}:A` });
  const b = h.start(argsB(), { session: `${h.session}:B` });
  const na = await h.notice(0); const nb = await h.notice(1);
  assert.notEqual(na.sessionKey, nb.sessionKey);
  h.decide(nb);
  assert.equal((await b.result).allowed, true);
  assert.equal(a.state, 'pending');
  h.decide(na);
  assert.equal((await a.result).allowed, true);
  assert.deepEqual([a.effects, b.effects], [1, 1]);
});

test('independent: denial and stale callback cannot cancel another same-method owner', { timeout: 4000 }, async (t) => {
  const h = harness(t); const gate = h.gate();
  h.onNotice = (_, index) => index === 0 ? gate.promise : true;
  const a = h.start(); const na = await h.notice(0);
  h.decide(na, 'deny');
  const b = h.start(argsB()); const nb = await h.notice(1);
  gate.resolve(false);
  noDisclosure(await a.result, h.notices);
  assert.equal(h.queue.resolveByToken(na.token, 'approve'), false);
  assert.equal(b.state, 'pending');
  h.decide(nb);
  assert.equal((await b.result).allowed, true);
  assert.deepEqual([a.effects, b.effects], [0, 1]);
});

test('independent: expired callback cannot act on a later same-method request', { timeout: 4000 }, async (t) => {
  const h = harness(t, { ttl: 80 }); const gate = h.gate();
  h.onNotice = (_, index) => index === 0 ? gate.promise : true;
  const a = h.start(); const na = await h.notice(0);
  // The callback is intentionally unresolved. The absolute lifetime must end A.
  noDisclosure(await a.result, h.notices);
  assert.equal(h.queue.resolveByToken(na.token, 'approve'), false);
  const b = h.start(argsB()); const nb = await h.notice(1);
  gate.resolve(false);
  await new Promise((done) => setImmediate(done));
  h.decide(nb);
  assert.equal((await b.result).allowed, true);
  assert.deepEqual([a.effects, b.effects], [0, 1]);
});

test('independent: approval cannot make a never-completing notification live forever', { timeout: 4000 }, async (t) => {
  const h = harness(t, { ttl: 80 }); const gate = h.gate();
  h.onNotice = () => gate.promise;
  const a = h.start(); const na = await h.notice(0); h.decide(na);
  noDisclosure(await a.result, h.notices);
  assert.equal(a.effects, 0);
  gate.resolve(true);
  assert.equal(h.queue.resolveByToken(na.token, 'approve'), false);
});

for (const mode of ['false', 'throw', 'approve_then_false']) {
  test(`independent: notification ${mode} denies only its owner`, { timeout: 4000 }, async (t) => {
    const h = harness(t);
    h.onNotice = (notice, index) => {
      if (index !== 0) return true;
      if (mode === 'approve_then_false') h.decide(notice);
      if (mode === 'throw') throw new Error(`Fake delivery failure ${notice.token}`);
      return false;
    };
    const a = h.start(); await h.notice(0);
    noDisclosure(await a.result, h.notices);
    const b = h.start(argsB()); const nb = await h.notice(1); h.decide(nb);
    assert.equal((await b.result).allowed, true);
    assert.deepEqual([a.effects, b.effects], [0, 1]);
  });
}

test('independent: nested argument mutation invalidates pending approval', { timeout: 4000 }, async (t) => {
  const h = harness(t); const args = argsA();
  const a = h.start(args); const na = await h.notice(0);
  args[0].options.append = true;
  h.decide(na);
  noDisclosure(await a.result, h.notices);
  assert.equal(a.effects, 0);
});

test('independent: equivalent object-key reordering still permits unchanged values', { timeout: 4000 }, async (t) => {
  const h = harness(t); const args = argsA();
  const a = h.start(args); const na = await h.notice(0);
  const encoding = args[0].options.encoding;
  delete args[0].options.encoding;
  args[0].options.encoding = encoding;
  h.decide(na);
  assert.equal((await a.result).allowed, true);
  assert.equal(a.effects, 1);
});

test('independent: argument mutation during asynchronous approval logging rejects', { timeout: 4000 }, async (t) => {
  const h = harness(t); const args = argsA();
  const entered = deferred(); const release = h.gate();
  const original = DecisionLog.append;
  let held = false;
  DecisionLog.append = async function (record) {
    if (record.decision === 'APPROVED' && !held) {
      held = true; entered.resolve(); await release.promise;
    }
    return original.call(this, record);
  };
  t.after(() => { DecisionLog.append = original; release.resolve(); });
  const a = h.start(args); const na = await h.notice(0); h.decide(na);
  await entered.promise;
  args[0].content = 'changed after the approval, during logging';
  release.resolve();
  noDisclosure(await a.result, h.notices);
  assert.equal(a.effects, 0);
});

test('independent: current hard policy denial overrides a pending human approval', { timeout: 4000 }, async (t) => {
  const h = harness(t);
  const a = h.start(); const na = await h.notice(0);
  h.policy.modules.FileSystem.write.action = 'DENY';
  h.decide(na);
  noDisclosure(await a.result, h.notices);
  assert.equal(a.effects, 0);
  assert.equal(h.notices.length, 1);
});

test('independent: changed ASK path scope requires a fresh invocation', { timeout: 4000 }, async (t) => {
  const policy = policyFor(); policy.modules.FileSystem.write.allowPaths = ['/synthetic/**'];
  const h = harness(t, { policy });
  const a = h.start(); const na = await h.notice(0);
  // The path remains permitted, but the presented scope has changed.
  policy.modules.FileSystem.write.allowPaths = ['/synthetic/reviewed.txt'];
  h.decide(na);
  noDisclosure(await a.result, h.notices);
  const b = h.start(); const nb = await h.notice(1); h.decide(nb);
  assert.equal((await b.result).allowed, true);
  assert.deepEqual([a.effects, b.effects], [0, 1]);
});

test('independent: unsupported accessors and proxies reject without invoking traps', { timeout: 4000 }, async (t) => {
  const h = harness(t); let reads = 0;
  const accessor = { path: '/synthetic/reviewed.txt' };
  Object.defineProperty(accessor, 'content', { enumerable: true, get() { reads += 1; return 'value'; } });
  const proxy = new Proxy({}, {
    get() { reads += 1; return undefined; },
    ownKeys() { reads += 1; return []; },
    getPrototypeOf() { reads += 1; return Object.prototype; },
    getOwnPropertyDescriptor() { reads += 1; return undefined; },
  });
  for (const argument of [accessor, proxy]) {
    const operation = h.start([argument]);
    assert.equal((await operation.result).allowed, false);
    assert.equal(operation.effects, 0);
  }
  assert.equal(reads, 0);
  assert.equal(h.notices.length, 0);
});
