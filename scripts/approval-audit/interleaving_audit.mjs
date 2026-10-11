import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import {
  existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync,
  realpathSync, rmSync, writeFileSync,
} from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { tmpdir } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const digest = (data) => createHash('sha256').update(data).digest('hex');
const fileHash = (path) => digest(readFileSync(path));
const canonical = (value) => JSON.stringify(value);

function saveNew(path, value) {
  if (existsSync(path)) throw new Error(`Refusing to overwrite ${path}`);
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

function allOrders() {
  const ids = ['A', 'B'];
  const names = ['create', 'first_wait', 'duplicate_wait', 'decide'];
  const events = ids.flatMap((id) => names.map((name) => `${id}.${name}`));
  const result = [];
  function visit(prefix, remaining) {
    if (!remaining.length) { result.push(prefix); return; }
    for (const event of remaining) {
      const [id, name] = event.split('.');
      if (name !== 'create' && !prefix.includes(`${id}.create`)) continue;
      if (name === 'duplicate_wait' && !prefix.includes(`${id}.first_wait`)) continue;
      visit([...prefix, event], remaining.filter((other) => other !== event));
    }
  }
  visit([], events);
  assert.equal(result.length, 630);
  assert.equal(new Set(result.map(canonical)).size, 630);
  for (const order of result) {
    assert.equal(new Set(order).size, 8);
    for (const id of ids) {
      for (const name of names.slice(1)) assert.ok(order.indexOf(`${id}.create`) < order.indexOf(`${id}.${name}`));
      assert.ok(order.indexOf(`${id}.first_wait`) < order.indexOf(`${id}.duplicate_wait`));
    }
  }
  return result;
}

function allocation() {
  const decisions = ['approve', 'deny', 'cancel'];
  return decisions.flatMap((A) => decisions.flatMap((B) =>
    allOrders().map((order) => ({ decisions: { A, B }, order }))));
}

function boundFiles(repo) {
  const paths = ['package.json', 'package-lock.json', 'src/core/ApprovalQueue.ts'];
  function walk(root) {
    for (const item of readdirSync(root, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(root, item.name);
      if (item.isSymbolicLink()) continue;
      if (item.isDirectory()) walk(path);
      else if (/\.(?:js|cjs|mjs|json|node)$/.test(path)) paths.push(relative(repo, path));
    }
  }
  walk(join(repo, 'dist'));
  walk(join(repo, 'node_modules'));
  return Object.fromEntries(paths.sort().map((path) => [path, fileHash(join(repo, path))]));
}

function runtime() {
  return { version: process.version, executable: realpathSync(process.execPath),
    executableSha256: fileHash(realpathSync(process.execPath)) };
}

const [command, repoArg, outputArg] = process.argv.slice(2);
if (command === 'self-test') {
  const cases = allocation();
  assert.equal(cases.length, 5670);
  assert.equal(new Set(cases.map(canonical)).size, 5670);
  console.log(JSON.stringify({ generator: 'passed', orders: 630, scenarios: 5670 }));
  process.exit(0);
}
assert.ok(['freeze', 'run'].includes(command), 'Use self-test, freeze REPO OUTPUT or run REPO OUTPUT');
const repo = resolve(repoArg);
const output = resolve(outputArg);
const sourceIdentity = () => ({
  driver: fileHash(fileURLToPath(import.meta.url)),
  protocol: fileHash(join(here, 'interleaving_protocol.md')),
});
if (command === 'freeze') {
  mkdirSync(output, { recursive: true });
  const identity = {
    version: 'approval-interleaving-v1', repo, runtime: runtime(), sources: sourceIdentity(),
    boundFiles: boundFiles(repo), allocationSha256: digest(canonical(allocation())),
    orders: 630, decisionPairs: 9, scenarios: 5670,
    scope: 'Bounded two-request queue order check; independent integration fixtures cover notifications, mutation and expiry.',
  };
  saveNew(join(output, 'freeze.json'), { identity, sha256: digest(canonical(identity)) });
  console.log(JSON.stringify({ status: 'frozen', scenarios: 5670 }));
  process.exit(0);
}

const frozen = JSON.parse(readFileSync(join(output, 'freeze.json'), 'utf8'));
assert.equal(frozen.sha256, digest(canonical(frozen.identity)));
assert.deepEqual(frozen.identity.runtime, runtime());
assert.deepEqual(frozen.identity.sources, sourceIdentity());
assert.deepEqual(frozen.identity.boundFiles, boundFiles(repo));
assert.equal(frozen.identity.allocationSha256, digest(canonical(allocation())));
assert.equal(frozen.identity.repo, repo);
assert.ok(!existsSync(join(output, 'results.json')), 'Run has already produced results');

const scratch = mkdtempSync(join(tmpdir(), 'reins-queue-order-'));
const oldHome = process.env.OPENCLAW_HOME;
process.env.OPENCLAW_HOME = scratch;
const require = createRequire(join(repo, 'package.json'));
const { logger } = require(join(repo, 'dist/core/Logger.js'));
const { ApprovalQueue } = require(join(repo, 'dist/core/ApprovalQueue.js'));
logger.silent = true;
const flush = () => new Promise((done) => setImmediate(done));
async function bounded(promise, message) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error(message)), 1000);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

// The reference machine has no dependency on queue implementation state.
function modelStep(model, event, decision) {
  const [, name] = event.split('.');
  if (name === 'create') model.state = 'pending';
  if (name === 'first_wait') {
    model.firstAttached = true;
    if (model.state === 'approved') { model.first = true; model.state = 'closed'; }
    else if (model.state === 'closed') model.first = false;
  }
  if (name === 'duplicate_wait') model.duplicate = false;
  if (name === 'decide') {
    if (decision === 'approve') {
      if (model.firstAttached) { model.first = true; model.state = 'closed'; }
      else model.state = 'approved';
    } else {
      model.state = 'closed';
      if (model.firstAttached) model.first = false;
    }
  }
}

const rows = [];
let failure;
try {
  for (const [index, scenario] of allocation().entries()) {
    const queue = new ApprovalQueue(120_000);
    const foreign = new ApprovalQueue(120_000);
    const contexts = {};
    const tokens = {};
    const observed = { A: { first: 'unattached', duplicate: 'unattached' }, B: { first: 'unattached', duplicate: 'unattached' } };
    const models = Object.fromEntries(['A', 'B'].map((id) => [id,
      { state: 'absent', firstAttached: false, first: 'unattached', duplicate: 'unattached' }]));
    const waiting = [];
    const snapshots = [];
    let scenarioError;
    try {
      for (const event of scenario.order) {
        const [id, name] = event.split('.');
        const model = models[id];
        if (name === 'create') {
          contexts[id] = queue.request('same-session', 'FileSystem', 'write', {
            binding: `independent-binding-${id}`, actionSummary: `Request ${id}`,
          });
          assert.ok(contexts[id]?.handle, 'Request must return an owning handle');
          tokens[id] = queue.getNotificationInfo(contexts[id].handle)?.token;
          assert.equal(typeof tokens[id], 'string');
          assert.equal(foreign.cancel(contexts[id].handle), false);
          assert.equal(foreign.getNotificationInfo(contexts[id].handle), undefined);
          assert.equal(await bounded(foreign.waitForApproval(contexts[id].handle), 'Foreign handle did not reject promptly'), false);
          assert.equal(queue.cancel({ ...contexts[id].handle }), false, 'Copied handle must not own the entry');
          if (tokens.A && tokens.B) assert.notEqual(tokens.A, tokens.B);
        } else if (name.endsWith('_wait')) {
          const kind = name === 'first_wait' ? 'first' : 'duplicate';
          observed[id][kind] = 'pending';
          if (kind === 'first') model.first = 'pending';
          waiting.push(queue.waitForApproval(contexts[id].handle).then((value) => {
            observed[id][kind] = value;
          }, (error) => { observed[id][kind] = { error: String(error) }; }));
        } else if (name === 'decide') {
          const result = scenario.decisions[id] === 'cancel'
            ? queue.cancel(contexts[id].handle)
            : queue.resolveByToken(tokens[id], scenario.decisions[id]);
          assert.equal(result, true, `First decision failed at ${event}`);
        }
        modelStep(model, event, scenario.decisions[id]);
        await flush();
        const expected = Object.fromEntries(['A', 'B'].map((key) => [key,
          { first: models[key].first, duplicate: models[key].duplicate }]));
        snapshots.push({ event, expected, observed: structuredClone(observed) });
        assert.deepEqual(observed, expected, `Settlement mismatch after ${event}`);
      }
      for (const id of ['A', 'B']) {
        assert.equal(queue.resolveByToken(tokens[id], 'approve'), false, 'Token replay must fail');
        assert.equal(queue.getNotificationInfo(contexts[id].handle), undefined);
        assert.equal(queue.cancel(contexts[id].handle), false, 'Settled entry remains live');
      }
    } catch (error) {
      scenarioError = { message: error.message, stack: error.stack, snapshots };
    } finally {
      for (const context of Object.values(contexts)) queue.cancel(context.handle);
      await flush();
      try {
        await bounded(Promise.all(waiting), 'Cleanup left a pending waiter');
      } catch (error) {
        scenarioError = { message: error.message, stack: error.stack, snapshots, previous: scenarioError };
      }
    }
    rows.push({ index, ...scenario, status: scenarioError ? 'failed' : 'passed', observed,
      snapshotsSha256: digest(canonical(snapshots)), ...(scenarioError ? { failure: scenarioError } : {}) });
    if (scenarioError) { failure = scenarioError; break; }
  }
} catch (error) {
  failure = { message: error.message, stack: error.stack };
} finally {
  logger.close();
  if (oldHome === undefined) delete process.env.OPENCLAW_HOME;
  else process.env.OPENCLAW_HOME = oldHome;
  rmSync(scratch, { recursive: true, force: true });
}
try {
  assert.deepEqual(frozen.identity.runtime, runtime());
  assert.deepEqual(frozen.identity.sources, sourceIdentity());
  assert.deepEqual(frozen.identity.boundFiles, boundFiles(repo));
} catch (error) {
  failure = { kind: 'runtime_or_source_identity_changed', message: error.message,
    previous: failure };
}
const result = {
  status: failure ? 'failed' : rows.length === 5670 ? 'complete_pass' : 'incomplete',
  freezeSha256: frozen.sha256, expectedScenarios: 5670, executedScenarios: rows.length,
  passed: rows.filter((row) => row.status === 'passed').length,
  failed: rows.filter((row) => row.status === 'failed').length,
  ...(failure ? { failure } : {}), rows,
};
saveNew(join(output, 'results.json'), result);
console.log(JSON.stringify({ ...result, rows: undefined, failure: failure?.message }));
process.exitCode = result.status === 'complete_pass' ? 0 : 1;
