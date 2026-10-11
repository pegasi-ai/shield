import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);

test('channel policy denials do not claim an approval request was sent', async (t) => {
  const scratch = mkdtempSync(join(tmpdir(), 'reins-channel-denial-'));
  const previousHome = process.env.OPENCLAW_HOME;
  const ttyDescriptor = Object.getOwnPropertyDescriptor(process.stdin, 'isTTY');
  process.env.OPENCLAW_HOME = scratch;
  Object.defineProperty(process.stdin, 'isTTY', { value: false, configurable: true });
  const { logger } = require('../dist/core/Logger.js');
  logger.silent = true;
  const { Interceptor } = require('../dist/core/Interceptor.js');
  t.after(() => {
    logger.close();
    if (previousHome === undefined) delete process.env.OPENCLAW_HOME;
    else process.env.OPENCLAW_HOME = previousHome;
    if (ttyDescriptor) Object.defineProperty(process.stdin, 'isTTY', ttyDescriptor);
    else delete process.stdin.isTTY;
    rmSync(scratch, { recursive: true, force: true });
  });

  const cases = [
    ['explicit denial', { action: 'DENY', description: 'Forbidden by policy' }, [{}]],
    ['default denial', null, [{}]],
    [
      'disallowed target',
      { action: 'ALLOW', allowPaths: ['https://allowed.test/**'] },
      [{ url: 'https://blocked.test/data' }],
    ],
    ['missing required target', { action: 'ALLOW', allowPaths: ['https://allowed.test/**'] }, [{}]],
  ];
  for (const [label, rule, args] of cases) {
    await t.test(label, async () => {
      const interceptor = new Interceptor(
        {
          defaultAction: 'DENY',
          modules: rule ? { Network: { fetch: rule } } : {},
        },
        false
      );
      let notifications = 0;
      let judgments = 0;
      let effects = 0;
      interceptor.onBlockCallback = async () => {
        notifications += 1;
        return true;
      };
      interceptor.arbitrator.judge = async () => {
        judgments += 1;
        return true;
      };
      await assert.rejects(
        async () => {
          await interceptor.evaluate('Network', 'fetch', args, 'channel-denial-test', {
            forceAsk: true,
            requiresExplicitConfirmation: true,
          });
          effects += 1;
        },
        (error) => {
          assert.match(error.message, /Reins Security Violation: Network\.fetch\(\) was DENIED/);
          assert.doesNotMatch(
            error.message,
            /APPROVAL_REQUIRED|approval request has been sent|pending human approval/
          );
          return true;
        }
      );
      assert.equal(notifications, 0);
      assert.equal(judgments, 0);
      assert.equal(effects, 0);
    });
  }

  await t.test('ASK retains the channel approval response', async () => {
    const interceptor = new Interceptor(
      {
        defaultAction: 'DENY',
        modules: { Network: { fetch: { action: 'ASK' } } },
      },
      false
    );
    let judgments = 0;
    interceptor.arbitrator.judge = async () => {
      judgments += 1;
      return false;
    };
    await assert.rejects(
      interceptor.evaluate('Network', 'fetch', [{}], 'channel-ask-test'),
      /\[Reins:APPROVAL_REQUIRED\]/
    );
    assert.equal(judgments, 1);
  });

  const decisions = readFileSync(join(scratch, 'reins', 'decisions.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(JSON.parse);
  assert.equal(decisions.filter(({ decision }) => decision === 'BLOCKED').length, 4);
  assert.equal(decisions.filter(({ decision }) => decision === 'REJECTED').length, 1);
});
