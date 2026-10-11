import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);

function trustedPolicy(recipient) {
  return {
    defaultAction: 'DENY',
    modules: {
      Network: { publish: { action: 'ALLOW', allowPaths: [`mailto:${recipient}`] } },
    },
  };
}

test('an explicitly installed current policy blocks a remembered revoked permission before delivery', async (t) => {
  const scratch = mkdtempSync(join(tmpdir(), 'reins-remembered-permission-'));
  const previousHome = process.env.OPENCLAW_HOME;
  process.env.OPENCLAW_HOME = scratch;
  const { logger } = require('../dist/core/Logger.js');
  logger.silent = true;
  const { Interceptor } = require('../dist/core/Interceptor.js');
  t.after(() => {
    logger.close();
    if (previousHome === undefined) delete process.env.OPENCLAW_HOME;
    else process.env.OPENCLAW_HOME = previousHome;
    rmSync(scratch, { recursive: true, force: true });
  });

  const oldRecipient = 'former-team@example.test';
  const currentRecipient = 'current-team@example.test';
  const report = 'Synthetic report. No external delivery.';
  const rememberedGrant = { policy_version: 'v1', recipient: oldRecipient, authorized: true };
  const memoryBefore = JSON.stringify(rememberedGrant);

  // The old grant was real. Evaluation alone creates no delivery effect.
  const previousPolicy = new Interceptor(trustedPolicy(oldRecipient), false);
  await previousPolicy.evaluate('Network', 'publish', [{ url: `mailto:${oldRecipient}` }]);

  // Trusted application code explicitly installs v2. This is not a hot-reload test.
  // Neither the remembered grant nor a prior ALLOW supplies the current policy.
  const currentPolicy = new Interceptor(trustedPolicy(currentRecipient), false);
  const deliveryFile = join(scratch, 'deliveries.json');
  writeFileSync(deliveryFile, '[]');
  const readDeliveries = () => JSON.parse(readFileSync(deliveryFile, 'utf8'));
  async function publishReport(recipient) {
    await currentPolicy.evaluate('Network', 'publish', [{ url: `mailto:${recipient}` }]);
    const deliveries = readDeliveries();
    deliveries.push({ recipient, report });
    writeFileSync(deliveryFile, JSON.stringify(deliveries));
  }

  await assert.rejects(publishReport(rememberedGrant.recipient), {
    message:
      'Reins Security Violation: Network.publish() was DENIED. ' +
      `Path "mailto:${oldRecipient}" is not in the allowed paths list`,
  });
  assert.deepEqual(readDeliveries(), [], 'a denied stale request must create no saved effect');
  assert.equal(
    JSON.stringify(rememberedGrant),
    memoryBefore,
    'memory still contains the old grant'
  );

  await publishReport(currentRecipient);
  assert.deepEqual(readDeliveries(), [{ recipient: currentRecipient, report }]);

  const decisions = readFileSync(join(scratch, 'reins', 'decisions.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(JSON.parse);
  assert.deepEqual(
    decisions.map(({ decision, args }) => ({ decision, args })),
    [
      { decision: 'ALLOWED', args: [{ url: `mailto:${oldRecipient}` }] },
      { decision: 'BLOCKED', args: [{ url: `mailto:${oldRecipient}` }] },
      { decision: 'ALLOWED', args: [{ url: `mailto:${currentRecipient}` }] },
    ]
  );
});
