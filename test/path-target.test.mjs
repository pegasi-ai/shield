import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);

test('configured path restrictions require a usable tool target before any effect', async (t) => {
  const scratch = mkdtempSync(join(tmpdir(), 'reins-path-target-'));
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

  const target = join(scratch, 'allowed.txt');
  const configurations = [
    ['FileSystem', 'write', 'path', target],
    ['Network', 'publish', 'url', 'mailto:current-team@example.test'],
    ['Browser', 'navigate', 'url', 'https://example.test/allowed'],
  ];
  let expectedBlocked = 0;
  for (const [moduleName, methodName, field, validTarget] of configurations) {
    for (const restriction of ['allowPaths', 'denyPaths']) {
      const rule = {
        action: 'ALLOW',
        [restriction]: [restriction === 'allowPaths' ? validTarget : '**/forbidden/**'],
      };
      const interceptor = new Interceptor(
        { defaultAction: 'DENY', modules: { [moduleName]: { [methodName]: rule } } },
        false
      );
      const invalidArguments = [
        [],
        [{}],
        [null],
        [[]],
        [{ [field]: '' }],
        [{ [field]: 42 }],
        [{ [field]: {} }],
      ];
      for (const [index, args] of invalidArguments.entries()) {
        await t.test(
          `${moduleName} ${restriction} rejects missing or invalid target ${index}`,
          async () => {
            const effects = [];
            await assert.rejects(async () => {
              await interceptor.evaluate(moduleName, methodName, args);
              effects.push('synthetic effect');
            }, /Path policy requires a non-empty string target/);
            assert.deepEqual(effects, []);
          }
        );
        expectedBlocked += 1;
      }
      await interceptor.evaluate(moduleName, methodName, [{ [field]: validTarget }]);
    }
  }

  // Existing aliases still supply the target; unrestricted rules do not need one.
  const filePolicy = new Interceptor(
    {
      defaultAction: 'DENY',
      modules: { FileSystem: { read: { action: 'ALLOW', allowPaths: [target] } } },
    },
    false
  );
  await filePolicy.evaluate('FileSystem', 'read', [{ file_path: target }]);
  await filePolicy.evaluate('FileSystem', 'read', [{ target }]);
  const networkPolicy = new Interceptor(
    {
      defaultAction: 'DENY',
      modules: {
        Network: { fetch: { action: 'ALLOW', allowPaths: ['https://example.test/allowed'] } },
      },
    },
    false
  );
  await networkPolicy.evaluate('Network', 'fetch', [{ src: 'https://example.test/allowed' }]);
  const unrestricted = new Interceptor({ defaultAction: 'ALLOW', modules: {} }, false);
  await unrestricted.evaluate('Other', 'run', [{}]);

  const decisions = readFileSync(join(scratch, 'reins', 'decisions.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map(JSON.parse);
  assert.equal(decisions.filter(({ decision }) => decision === 'BLOCKED').length, expectedBlocked);
  assert.equal(decisions.filter(({ decision }) => decision === 'ALLOWED').length, 10);
});
