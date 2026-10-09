#!/usr/bin/env node
'use strict';

// This file and the Reins installation belong to the trusted custom-tool side.
// Student input can choose a recipient, never a policy, module, or method.
const fs = require('node:fs');
const path = require('node:path');
const { TextDecoder } = require('node:util');

const MAX_INPUT_BYTES = 4096;
const POLICY = Object.freeze({
  defaultAction: 'DENY',
  modules: Object.freeze({
    Network: Object.freeze({
      publish: Object.freeze({
        action: 'ALLOW',
        allowPaths: Object.freeze(['mailto:team@example.test']),
      }),
    }),
  }),
});

class InputError extends Error {}

async function readRecipient() {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of process.stdin) {
    bytes += chunk.length;
    if (bytes > MAX_INPUT_BYTES) {
      throw new InputError('Request exceeds the 4096-byte limit.');
    }
    chunks.push(chunk);
  }

  let request;
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks));
    request = JSON.parse(text);
  } catch {
    throw new InputError('Request must be one valid UTF-8 JSON object.');
  }

  if (
    request === null ||
    typeof request !== 'object' ||
    Array.isArray(request) ||
    Object.getPrototypeOf(request) !== Object.prototype ||
    Object.keys(request).length !== 1 ||
    !Object.hasOwn(request, 'recipient') ||
    typeof request.recipient !== 'string'
  ) {
    throw new InputError('Request must contain only a recipient string.');
  }

  // A deliberately small mailbox syntax for this example. Reject URL parameters,
  // control characters, display names, and multiple recipients before evaluation.
  const recipient = request.recipient;
  if (
    recipient.length > 254 ||
    !/^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}$/.test(recipient)
  ) {
    throw new InputError('Recipient must be a single plain email address.');
  }
  return recipient;
}

function loadInterceptor() {
  const localEngine = path.resolve(__dirname, '../../dist/core/Interceptor.js');
  const enginePath = fs.existsSync(localEngine)
    ? localEngine
    : require.resolve('@pegasi-ai/reins/dist/core/Interceptor.js', { paths: [__dirname] });

  // Keep stdout a one-line protocol. Reins still writes decision records through
  // DecisionLog; this silences the Winston logger, not policy evaluation.
  const { logger } = require(path.join(path.dirname(enginePath), 'Logger.js'));
  logger.silent = true;
  return require(enginePath).Interceptor;
}

async function main() {
  try {
    const recipient = await readRecipient();
    const Interceptor = loadInterceptor();
    const interceptor = new Interceptor(POLICY, false);
    const target = `mailto:${recipient}`;
    try {
      await interceptor.evaluate('Network', 'publish', [{ url: target }]);
    } catch (error) {
      // Reins currently exposes generic errors, so recognize only the precise
      // denial this fixed policy can produce. All other failures are ERROR.
      const expectedDenial =
        'Reins Security Violation: Network.publish() was DENIED. ' +
        `Path "${target}" is not in the allowed paths list`;
      if (error instanceof Error && error.message === expectedDenial) {
        emit({ decision: 'DENY', reason: 'Recipient is outside the trusted team allowlist.' });
        return;
      }
      throw error;
    }
    emit({ decision: 'ALLOW', reason: 'Recipient is permitted by the trusted Reins policy.' });
  } catch (error) {
    emit({
      decision: 'ERROR',
      reason:
        error instanceof InputError ? error.message : 'Reins policy engine unavailable or failed.',
    });
    process.exitCode = 1;
  }
}

function emit(result) {
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

void main();
