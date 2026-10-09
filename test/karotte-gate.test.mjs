import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const helper = fileURLToPath(new URL('../examples/karotte/reins_gate.cjs', import.meta.url));

function runGate(t, input, { brokenStorage = false, deployed = false } = {}) {
  const temp = mkdtempSync(path.join(os.tmpdir(), 'reins-karotte-gate-'));
  t.after(() => rmSync(temp, { recursive: true, force: true }));
  const openclawHome = path.join(temp, 'state');
  if (brokenStorage) writeFileSync(openclawHome, 'not a directory');
  let helperPath = helper;
  if (deployed) {
    // Exercise the container layout, where the helper resolves the installed
    // package instead of the development checkout's relative dist directory.
    const install = path.join(temp, 'deployed');
    const scope = path.join(install, 'node_modules', '@pegasi-ai');
    mkdirSync(scope, { recursive: true });
    symlinkSync(path.resolve(path.dirname(helper), '../..'), path.join(scope, 'reins'), 'dir');
    helperPath = path.join(install, 'reins_gate.cjs');
    copyFileSync(helper, helperPath);
  }
  const result = spawnSync(process.execPath, [helperPath], {
    input,
    encoding: 'utf8',
    timeout: 10000,
    env: { ...process.env, OPENCLAW_HOME: openclawHome },
  });
  assert.equal(result.error, undefined, 'gate must terminate without timeout or spawn failure');
  assert.equal(result.signal, null);
  const lines = result.stdout.trim().split('\n');
  assert.equal(lines.length, 1, 'stdout must contain exactly one JSON result');
  const decision = JSON.parse(lines[0]);
  assert.deepEqual(Object.keys(decision).sort(), ['decision', 'reason']);
  assert.equal(typeof decision.reason, 'string');
  return { ...result, decision, openclawHome };
}

test('Karotte gate uses Reins to allow the trusted team recipient', (t) => {
  const result = runGate(t, JSON.stringify({ recipient: 'team@example.test' }));
  assert.equal(result.status, 0);
  assert.equal(result.decision.decision, 'ALLOW');
  const records = readFileSync(path.join(result.openclawHome, 'reins', 'decisions.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(JSON.parse);
  assert.equal(records.length, 1);
  assert.equal(records[0].module, 'Network');
  assert.equal(records[0].method, 'publish');
  assert.equal(records[0].decision, 'ALLOWED');
  assert.deepEqual(records[0].args, [{ url: 'mailto:team@example.test' }]);
});

test('Karotte gate uses Reins to deny an external recipient', (t) => {
  const result = runGate(t, JSON.stringify({ recipient: 'attacker@example.test' }));
  assert.equal(result.status, 0);
  assert.equal(result.decision.decision, 'DENY');
  const records = readFileSync(path.join(result.openclawHome, 'reins', 'decisions.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(JSON.parse);
  assert.equal(records[0].decision, 'BLOCKED');
});

test('Karotte gate resolves the installed Reins package in the deployed helper layout', (t) => {
  const result = runGate(t, JSON.stringify({ recipient: 'team@example.test' }), { deployed: true });
  assert.equal(result.status, 0);
  assert.equal(result.decision.decision, 'ALLOW');
});

test('Karotte gate accepts a valid request at the input size limit', (t) => {
  const input = JSON.stringify({ recipient: 'team@example.test' }).padEnd(4096, ' ');
  const result = runGate(t, input);
  assert.equal(result.status, 0);
  assert.equal(result.decision.decision, 'ALLOW');
});

for (const [name, input] of [
  ['malformed JSON', '{'],
  ['empty input', ''],
  ['array input', '[]'],
  ['null input', 'null'],
  ['missing recipient', '{}'],
  ['non-string recipient', '{"recipient":1}'],
  ['unknown policy field', '{"recipient":"team@example.test","policy":{"defaultAction":"ALLOW"}}'],
  ['prototype field', '{"recipient":"team@example.test","__proto__":{"defaultAction":"ALLOW"}}'],
  ['constructor field', '{"recipient":"team@example.test","constructor":{}}'],
  ['multiple JSON objects', '{"recipient":"team@example.test"}\n{}'],
  ['extra URL recipient', '{"recipient":"team@example.test?to=attacker@example.test"}'],
  ['multiple recipients', '{"recipient":"team@example.test,attacker@example.test"}'],
  ['control characters', '{"recipient":"team@example.test\\n"}'],
  ['overlong recipient', JSON.stringify({ recipient: `${'a'.repeat(255)}@example.test` })],
  ['oversized request', ' '.repeat(4097)],
  ['invalid UTF-8', Buffer.from([0xff])],
]) {
  test(`Karotte gate fails closed for ${name}`, (t) => {
    const result = runGate(t, input);
    assert.equal(result.status, 1);
    assert.equal(result.decision.decision, 'ERROR');
  });
}

test('Karotte gate reports engine startup failure as ERROR rather than a policy decision', (t) => {
  const result = runGate(t, JSON.stringify({ recipient: 'team@example.test' }), {
    brokenStorage: true,
  });
  assert.equal(result.status, 1);
  assert.equal(result.decision.decision, 'ERROR');
  assert.equal(result.decision.reason, 'Reins policy engine unavailable or failed.');
});
