/**
 * Reins Interceptor
 * The Brain - Runtime Security Evaluation Engine
 */

import crypto from 'crypto';
import path from 'path';
import {
  SecurityPolicy,
  Decision,
  SecurityRule,
  ExecutionContext,
  InterventionMetadata,
  ApprovalCallback,
} from '../types';
import { DEFAULT_POLICY } from '../config';
import { Arbitrator } from './Arbitrator';
import { bindApprovalArguments, ApprovalBinding } from './ApprovalBinding';
import { DecisionLog, DecisionRecord } from '../storage/DecisionLog';
import { StatsTracker } from '../storage/StatsTracker';
import { trustRateLimiter } from './TrustRateLimiter';
import { logger } from './Logger';
import chalk from 'chalk';

// ---------------------------------------------------------------------------
// Path-policy helpers (no external dependency)
// Supports: * (within a segment), ** (any depth), ? (single char)
// ---------------------------------------------------------------------------

function expandTilde(raw: string): string {
  if (raw === '~' || raw.startsWith('~/') || raw.startsWith('~\\')) {
    const home = process.env.HOME ?? process.env.USERPROFILE ?? '~';
    return home + raw.slice(1);
  }
  return raw;
}

function matchesGlob(pattern: string, str: string): boolean {
  const norm = str.replace(/\\/g, '/');
  const pat = expandTilde(pattern).replace(/\\/g, '/');

  const regex = pat
    .replace(/[.+^${}()|[\]]/g, '\\$&') // escape regex specials
    .replace(/\*\*/g, '\x00')            // placeholder for **
    .replace(/\*/g, '[^/]*')             // * → any non-slash chars
    .replace(/\?/g, '[^/]')              // ? → single non-slash char
    .replace(/\/\x00/g, '(?:/.+)?')     // /** → slash + content optional (matches dir root)
    .replace(/\x00/g, '.*');             // standalone ** → anything

  return new RegExp(`^${regex}$`).test(norm);
}

/**
 * Extract the path or URL that a tool call is targeting, for path-policy evaluation.
 * Returns null if no usable string target can be determined.
 */
function extractTarget(moduleName: string, params: Record<string, unknown>): string | null {
  const isUrl = moduleName === 'Browser' || moduleName === 'Network';
  const raw = isUrl
    ? (params.url ?? params.src)
    : (params.path ?? params.file_path ?? params.target);
  if (typeof raw !== 'string' || raw.length === 0) return null;
  if (isUrl) return raw;
  // Expand ~ before resolving so ~/.ssh/id_rsa becomes /home/user/.ssh/id_rsa
  return path.resolve(expandTilde(raw));
}

/**
 * Apply allowPaths / denyPaths from the rule against the actual target.
 * Returns a DENY rule if the path is out of bounds, otherwise returns the rule unchanged.
 */
function applyPathPolicy(rule: SecurityRule, moduleName: string, args: unknown[]): SecurityRule {
  if (!rule.allowPaths?.length && !rule.denyPaths?.length) return rule;

  const first = args[0];
  const params =
    first !== null && typeof first === 'object' && !Array.isArray(first)
      ? (first as Record<string, unknown>)
      : {};
  const target = extractTarget(moduleName, params);

  if (!target) {
    return {
      action: 'DENY',
      description: 'Path policy requires a non-empty string target',
    };
  }

  // denyPaths take precedence over allowPaths.
  if (rule.denyPaths?.some((p) => matchesGlob(p, target))) {
    return {
      action: 'DENY',
      description: `Path "${target}" matches a denied pattern`,
    };
  }

  // If allowPaths is set the target must match at least one entry.
  if (rule.allowPaths?.length && !rule.allowPaths.some((p) => matchesGlob(p, target))) {
    return {
      action: 'DENY',
      description: `Path "${target}" is not in the allowed paths list`,
    };
  }

  return rule;
}

// ---------------------------------------------------------------------------

export class Interceptor {
  private policy: SecurityPolicy;
  private arbitrator: Arbitrator;
  private logEnabled: boolean;

  /**
   * Trusted callback awaited within the approval deadline in channel mode.
   * Its fourth argument identifies the exact invocation for the notifier.
   */
  public onBlockCallback?: ApprovalCallback;

  constructor(policy?: SecurityPolicy, logEnabled: boolean = true) {
    this.policy = policy || DEFAULT_POLICY;
    this.arbitrator = new Arbitrator();
    this.logEnabled = logEnabled;
  }

  /**
   * Evaluate the security policy for a tool call.
   */
  async evaluate(
    moduleName: string,
    methodName: string,
    args: unknown[],
    sessionKey?: string,
    intervention?: InterventionMetadata
  ): Promise<void> {
    const baseRule = this.lookupRule(moduleName, methodName);
    // Bind before target extraction so unsupported ASK args cannot execute getters.
    const mayAskInChannel =
      !process.stdin.isTTY &&
      !!sessionKey &&
      (baseRule.action === 'ASK' || (baseRule.action === 'ALLOW' && !!intervention?.forceAsk));
    let argumentBinding: ApprovalBinding | undefined;
    if (mayAskInChannel) {
      try {
        argumentBinding = bindApprovalArguments(args);
      } catch {
        await this.logDecision({
          timestamp: new Date().toISOString(),
          module: moduleName,
          method: methodName,
          args: [],
          decision: 'REJECTED',
          decisionTime: 0,
          reason: 'Unsupported channel approval arguments (plain JSON, depth 64, 1 MiB required)',
        });
        throw new Error(
          `Reins Security Violation: ${moduleName}.${methodName}() has invalid channel approval arguments`
        );
      }
    }
    const checkedArgs = argumentBinding ? (argumentBinding.snapshot as unknown[]) : args;
    // Path-policy check runs before intervention so a denied path is never
    // upgradeable to ASK — it's a hard DENY.
    const originalRule = applyPathPolicy(baseRule, moduleName, checkedArgs);

    const normalizedIntervention = this.normalizeIntervention(
      moduleName,
      methodName,
      originalRule,
      intervention
    );
    const effectiveRule = this.applyInterventionToRule(
      originalRule,
      moduleName,
      methodName,
      normalizedIntervention
    );

    let approvalGuard: (() => boolean) | undefined;
    let approvalBinding: string | undefined;
    if (argumentBinding && effectiveRule.action === 'ASK') {
      const policyBinding = this.approvalPolicyBinding(
        baseRule,
        effectiveRule,
        normalizedIntervention
      );
      const boundArgs = argumentBinding.canonical;
      approvalBinding = crypto
        .createHash('sha256')
        .update(boundArgs)
        .update('\n')
        .update(policyBinding)
        .digest('hex');
      approvalGuard = (): boolean => {
        try {
          if (bindApprovalArguments(args).canonical !== boundArgs) return false;
          const currentBase = this.lookupRule(moduleName, methodName);
          const currentOriginal = applyPathPolicy(currentBase, moduleName, checkedArgs);
          const currentIntervention = this.normalizeIntervention(
            moduleName,
            methodName,
            currentOriginal,
            intervention
          );
          const currentEffective = this.applyInterventionToRule(
            currentOriginal,
            moduleName,
            methodName,
            currentIntervention
          );
          return (
            currentEffective.action === 'ASK' &&
            this.approvalPolicyBinding(currentBase, currentEffective, currentIntervention) ===
              policyBinding
          );
        } catch {
          return false;
        }
      };
    }

    if (this.logEnabled) {
      this.logInterception(moduleName, methodName, effectiveRule.action);
    }

    let allowed = await this.executeDecision(
      effectiveRule,
      moduleName,
      methodName,
      checkedArgs,
      sessionKey,
      normalizedIntervention,
      approvalGuard,
      approvalBinding
    );

    // This is the last successful-path check after every asynchronous decision/log step.
    if (allowed && approvalGuard && !approvalGuard()) {
      allowed = false;
      await this.logDecision({
        timestamp: new Date().toISOString(),
        module: moduleName,
        method: methodName,
        args: checkedArgs,
        decision: 'REJECTED',
        decisionTime: 0,
        reason: 'Channel approval invalidated: arguments or current policy changed',
      });
    }

    if (!allowed) {
      const isChannelMode = !process.stdin.isTTY && sessionKey;
      const detail = effectiveRule.description || 'No description provided.';

      if (isChannelMode && effectiveRule.action === 'ASK') {
        const instructions = this.buildChannelInstructions(
          moduleName,
          methodName,
          normalizedIntervention
        );

        throw new Error(
          `[Reins:APPROVAL_REQUIRED] ${moduleName}.${methodName}() was blocked without a usable invocation approval. ` +
            `Risk: ${detail}\n` +
            instructions
        );
      }

      throw new Error(
        `Reins Security Violation: ${moduleName}.${methodName}() was DENIED. ${detail}`
      );
    }
  }

  private approvalPolicyBinding(
    baseRule: SecurityRule,
    effectiveRule: SecurityRule,
    intervention?: InterventionMetadata
  ): string {
    const defined = (value: object): Record<string, unknown> =>
      Object.fromEntries(
        Object.entries(value).filter(
          ([key, item]) => item !== undefined && key !== 'confirmationToken'
        )
      );
    return bindApprovalArguments({
      baseRule: defined(baseRule),
      effectiveRule: defined(effectiveRule),
      intervention: defined(intervention || {}),
    }).canonical;
  }

  private normalizeIntervention(
    moduleName: string,
    methodName: string,
    rule: SecurityRule,
    intervention?: InterventionMetadata
  ): InterventionMetadata | undefined {
    if (!intervention) return undefined;

    const next: InterventionMetadata = { ...intervention };

    if (next.requiresExplicitConfirmation && !next.confirmationToken) {
      next.confirmationToken = this.generateConfirmationToken(moduleName, methodName);
    }

    if (next.overrideDescription && next.interventionReason) {
      next.overrideDescription = `${next.overrideDescription} ${next.interventionReason}`;
    } else if (!next.overrideDescription && next.interventionReason) {
      next.overrideDescription = next.interventionReason;
    }

    // No need to force ASK if policy is already DENY.
    if (rule.action === 'DENY') {
      next.forceAsk = false;
      next.requiresExplicitConfirmation = false;
    }

    return next;
  }

  private generateConfirmationToken(moduleName: string, methodName: string): string {
    const seed = `${moduleName}.${methodName}:${Date.now()}:${Math.random()}`;
    const digest = crypto.createHash('sha1').update(seed).digest('hex').slice(0, 6).toUpperCase();
    return `CONFIRM-${digest}`;
  }

  private applyInterventionToRule(
    rule: SecurityRule,
    moduleName: string,
    methodName: string,
    intervention?: InterventionMetadata
  ): SecurityRule {
    if (!intervention) {
      return rule;
    }

    const description =
      intervention.overrideDescription ||
      rule.description ||
      `Rule evaluation for ${moduleName}.${methodName}`;

    if (rule.action === 'DENY') {
      return { ...rule, description };
    }

    if (intervention.forceAsk && rule.action === 'ALLOW') {
      return {
        action: 'ASK',
        description,
      };
    }

    if (rule.action === 'ASK' && description !== rule.description) {
      return {
        ...rule,
        description,
      };
    }

    return rule;
  }

  private buildChannelInstructions(
    moduleName: string,
    methodName: string,
    intervention?: InterventionMetadata
  ): string {
    const strict = intervention?.requiresExplicitConfirmation === true;
    const summary = intervention?.actionSummary || `${moduleName}.${methodName}()`;

    const screenshotInstruction = intervention?.recommendScreenshotReview
      ? 'Before asking for approval, call screenshot() and use existing vision reasoning on the image. ' +
        'If still uncertain, escalate with the screenshot in the user message.\n'
      : '';

    if (strict) {
      return (
        `${screenshotInstruction}` +
        `⚠️ HIGH-RISK action requires explicit human confirmation.\n` +
        `Action: ${summary}\n` +
        `This invocation did not receive a usable approval. A new call requires its own approval.\n` +
        `Do NOT attempt to self-approve.`
      );
    }

    return (
      `${screenshotInstruction}` +
      `Action: ${summary}\n` +
      `This invocation did not receive a usable approval. A new call requires its own approval.\n` +
      `Do NOT attempt to self-approve.`
    );
  }

  private lookupRule(moduleName: string, methodName: string): SecurityRule {
    const moduleRules = this.policy.modules[moduleName];

    if (moduleRules && moduleRules[methodName]) {
      return moduleRules[methodName];
    }

    return {
      action: this.policy.defaultAction,
      description: `No specific rule defined for ${moduleName}.${methodName}`,
    };
  }

  private async executeDecision(
    rule: SecurityRule,
    moduleName: string,
    methodName: string,
    args: unknown[],
    sessionKey?: string,
    intervention?: InterventionMetadata,
    approvalGuard?: () => boolean,
    approvalBinding?: string
  ): Promise<boolean> {
    const startTime = Date.now();

    switch (rule.action) {
      case 'ALLOW': {
        const decisionTime = Date.now() - startTime;
        await this.logDecision({
          timestamp: new Date().toISOString(),
          module: moduleName,
          method: methodName,
          args,
          decision: 'ALLOWED',
          reason: rule.description,
          decisionTime,
        });
        return true;
      }

      case 'DENY': {
        const decisionTime = Date.now() - startTime;
        await this.logDecision({
          timestamp: new Date().toISOString(),
          module: moduleName,
          method: methodName,
          args,
          decision: 'BLOCKED',
          reason: rule.description || 'Policy: DENY',
          decisionTime,
        });
        return false;
      }

      case 'ASK': {
        const context: ExecutionContext = {
          moduleName,
          methodName,
          args,
          rule,
          sessionKey,
          intervention,
          onBlockCallback: this.onBlockCallback,
          approvalBinding,
        };

        let approved = await this.arbitrator.judge(context);
        const invalidated = approved && approvalGuard && !approvalGuard();
        if (invalidated) approved = false;
        const decisionTime = Date.now() - startTime;

        if (!approved && process.stdin.isTTY) {
          // TTY: record denial for cooldown escalation.
          trustRateLimiter.recordDenial(sessionKey || 'tty');
        }
        // Channel mode: onBlockCallback is called inside judgeChannel() before stalling.

        await this.logDecision({
          timestamp: new Date().toISOString(),
          module: moduleName,
          method: methodName,
          args,
          decision: approved ? 'APPROVED' : 'REJECTED',
          userId: 'human',
          reason: invalidated
            ? 'Channel approval invalidated: arguments or current policy changed'
            : rule.description,
          decisionTime,
        });

        return approved;
      }

      default:
        throw new Error(`Unknown decision type: ${rule.action}`);
    }
  }

  private async logDecision(record: DecisionRecord): Promise<void> {
    try {
      await DecisionLog.append(record);
      await StatsTracker.increment(record.decision, record.decisionTime);
    } catch (error) {
      logger.error('Failed to log decision', { error });
    }
  }

  private logInterception(moduleName: string, methodName: string, action: Decision): void {
    const coloredAction =
      action === 'ALLOW'
        ? chalk.green(action)
        : action === 'DENY'
          ? chalk.red(action)
          : chalk.yellow(action);

    logger.info(`${chalk.cyan('Reins:')} ${moduleName}.${methodName}() → ${coloredAction}`);
  }
}
