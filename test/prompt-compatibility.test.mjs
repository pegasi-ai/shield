import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));

function run(t, script, input) {
  const testRoot = mkdtempSync(path.join(os.tmpdir(), 'reins-prompt-compat-'));
  t.after(() => rmSync(testRoot, { recursive: true, force: true }));
  return spawnSync(process.execPath, ['-e', script], {
    cwd: root,
    env: { ...process.env, OPENCLAW_HOME: testRoot, LOG_LEVEL: 'error' },
    encoding: 'utf8',
    timeout: 5000,
    input,
  });
}

test('CommonJS package entry loads without opening the ESM prompt dependency', (t) => {
  const result = run(t, `
    const assert = require('node:assert/strict');
    const reins = require('./');
    assert.equal(typeof reins.Interceptor, 'function');
    assert.equal(typeof reins.Arbitrator, 'function');
  `);
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr);
});

test('lazy prompt loads real Inquirer and preserves initial answers', (t) => {
  const result = run(t, `
    const assert = require('node:assert/strict');
    const { prompt } = require('./dist/lib/prompt.js');
    prompt([], { verified: 'retained' }).then(answer => {
      assert.deepEqual(answer, { verified: 'retained' });
      console.log('INITIAL_ANSWERS_RETAINED');
    }).catch(error => { console.error(error); process.exitCode = 1; });
  `);
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /INITIAL_ANSWERS_RETAINED/);
});

test('lazy prompt accepts a real stdin answer', (t) => {
  const result = run(t, `
    const assert = require('node:assert/strict');
    const { prompt } = require('./dist/lib/prompt.js');
    prompt([{ type: 'input', name: 'value', message: 'Synthetic prompt:' }]).then(answer => {
      assert.equal(answer.value, 'independent reply');
      console.log('STDIN_ANSWER_RETAINED');
    }).catch(error => { console.error(error); process.exitCode = 1; });
  `, 'independent reply\n');
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /STDIN_ANSWER_RETAINED/);
});
