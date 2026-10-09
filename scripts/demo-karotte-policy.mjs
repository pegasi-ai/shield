#!/usr/bin/env node
// Local policy smoke check. The Karotte replay separately tests the sandbox path.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const helper = fileURLToPath(new URL('../examples/karotte/reins_gate.cjs', import.meta.url));
const scratch = mkdtempSync(join(tmpdir(), 'reins-karotte-smoke-'));
const deliveryFile = join(scratch, 'deliveries.json');
const deliveries = [];
try {
  console.log('Reins + Karotte | local policy smoke check');
  console.log('Real Reins decisions. Simulated delivery. No model or sandbox.\n');
  for (const [recipient, expected] of [
    ['outside@example.test', 'DENY'],
    ['team@example.test', 'ALLOW'],
  ]) {
    const result = spawnSync(process.execPath, [helper], {
      input: JSON.stringify({ recipient }),
      encoding: 'utf8',
      timeout: 10000,
      maxBuffer: 65536,
      env: { ...process.env, OPENCLAW_HOME: scratch },
    });
    assert.equal(result.error, undefined, 'Policy process must finish');
    assert.equal(result.status, 0, `Policy process failed: ${result.stderr}`);
    const decision = JSON.parse(result.stdout);
    assert.equal(decision.decision, expected);
    if (decision.decision === 'ALLOW') {
      deliveries.push({ recipient, report: 'weekly-summary' });
      writeFileSync(deliveryFile, JSON.stringify(deliveries));
    }
    console.log(
      `${decision.decision.padEnd(5)}  ${recipient.padEnd(24)} executed=${decision.decision === 'ALLOW'}`
    );
  }
  assert.deepEqual(JSON.parse(readFileSync(deliveryFile, 'utf8')), [
    { recipient: 'team@example.test', report: 'weekly-summary' },
  ]);
  console.log('\nVerified saved deliveries: 1 authorized, 0 unauthorized');
  console.log('Run npm run demo:karotte to exercise Karotte tools and grading.');
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
