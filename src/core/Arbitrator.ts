/**
 * Reins Arbitrator
 * The UI/Prompt Logic for Human-in-the-Loop Decisions
 *
 * Modes:
 *  1. TTY (interactive terminal)      → inquirer prompt
 *  2. Daemon + sessionKey (channel)   → owned invocation approval (wait under a fixed deadline)
 *  3. Daemon without sessionKey       → auto-deny (fail-secure)
 */

import { prompt } from '../lib/prompt';
import chalk from 'chalk';
import crypto from 'crypto';
import { ExecutionContext } from '../types';
import { logger } from './Logger';
import { approvalQueue, ApprovalQueue } from './ApprovalQueue';
import { bindApprovalArguments } from './ApprovalBinding';
import { DecisionLog } from '../storage/DecisionLog';

export class Arbitrator {
  constructor(private readonly queue: ApprovalQueue = approvalQueue) {}

  async judge(context: ExecutionContext): Promise<boolean> {
    if (process.stdin.isTTY) {
      return this.judgeTTY(context);
    }

    if (context.sessionKey) {
      return this.judgeChannel(context);
    }

    logger.info(
      `ASK policy → auto-denied (no TTY, no session): ${context.moduleName}.${context.methodName}()`,
      { args: context.args }
    );
    return false;
  }

  private async judgeTTY(context: ExecutionContext): Promise<boolean> {
    this.displayBanner();
    this.displayContext(context);

    if (context.intervention?.requiresExplicitConfirmation) {
      return this.judgeTTYExplicitConfirmation(context);
    }

    const answer = await prompt([
      {
        type: 'list',
        name: 'decision',
        message: chalk.bold.yellow('⚠️  What should Reins do?'),
        choices: [
          {
            name: chalk.green('✓ Approve - Allow this action'),
            value: true,
          },
          {
            name: chalk.red('✗ Reject - Block this action'),
            value: false,
          },
        ],
        default: 1,
      },
    ]);

    console.log('');

    if (answer.decision) {
      console.log(chalk.green('✓ Action APPROVED by user\n'));
    } else {
      console.log(chalk.red('✗ Action REJECTED by user\n'));
    }

    await this.logApprovalDecision(context, answer.decision ? 'yes' : 'no', answer.decision);

    return answer.decision;
  }

  private async judgeTTYExplicitConfirmation(context: ExecutionContext): Promise<boolean> {
    const token = context.intervention?.confirmationToken || 'CONFIRM';
    const summary =
      context.intervention?.actionSummary || `${context.moduleName}.${context.methodName}()`;

    console.log(chalk.bold.yellow('🔐 Irreversible action requires explicit confirmation.'));
    console.log(chalk.bold.cyan('Action Summary:'), chalk.white(summary));
    console.log(chalk.bold.cyan('Confirmation Token:'), chalk.white(token));
    console.log('');

    const answer = await prompt([
      {
        type: 'input',
        name: 'typed',
        message: `Type ${token} to approve, or anything else to reject:`,
      },
    ]);

    const approved =
      String(answer.typed || '')
        .trim()
        .toUpperCase() === token.toUpperCase();

    console.log('');
    if (approved) {
      console.log(chalk.green('✓ Action APPROVED by explicit confirmation\n'));
    } else {
      console.log(chalk.red('✗ Action REJECTED (confirmation token mismatch)\n'));
    }

    await this.logApprovalDecision(context, 'confirm', approved, String(answer.typed || ''));

    return approved;
  }

  private async judgeChannel(context: ExecutionContext): Promise<boolean> {
    const { sessionKey, moduleName, methodName } = context;
    let canonical: string;
    try {
      canonical = bindApprovalArguments(context.args).canonical;
    } catch {
      return false;
    }
    const request = this.queue.request(sessionKey!, moduleName, methodName, {
      binding:
        context.approvalBinding || crypto.createHash('sha256').update(canonical).digest('hex'),
      requiresExplicitConfirmation: context.intervention?.requiresExplicitConfirmation === true,
      actionSummary: context.intervention?.actionSummary,
    });
    if (!request) return false;
    try {
      if (!context.onBlockCallback) return false;
      const notified = await this.queue.notify(request.handle, () =>
        context.onBlockCallback!(sessionKey!, moduleName, methodName, request)
      );
      if (!notified) return false;
      const approved = await this.queue.waitForApproval(request.handle);
      // evaluate() also rechecks the caller's original arguments and current policy.
      return approved && bindApprovalArguments(context.args).canonical === canonical;
    } catch {
      return false;
    } finally {
      this.queue.cancel(request.handle);
    }
  }

  private displayBanner(): void {
    console.log('');
    console.log(chalk.bgRed.white.bold('═'.repeat(80)));
    console.log(
      chalk.bgRed.white.bold('   🪢 REINS SECURITY ALERT - HUMAN AUTHORIZATION REQUIRED')
    );
    console.log(chalk.bgRed.white.bold('═'.repeat(80)));
    console.log('');
  }

  private displayContext(context: ExecutionContext): void {
    if (context.intervention?.cooldownLevel) {
      const level = context.intervention.cooldownLevel;
      const label = level >= 2 ? 'RESTRICTED' : 'HEIGHTENED';
      console.log(
        chalk.bgYellow.black.bold(
          `   COOLDOWN ACTIVE: Level ${level} (${label}) — repeated denials detected   `
        )
      );
      console.log('');
    }

    console.log(chalk.bold.cyan('📦 Module:'), chalk.white(context.moduleName));
    console.log(chalk.bold.cyan('🔧 Method:'), chalk.white(context.methodName));

    if (context.rule.description) {
      console.log(chalk.bold.cyan('⚠️  Risk:'), chalk.yellow(context.rule.description));
    }
    if (context.intervention?.actionSummary) {
      console.log(chalk.bold.cyan('🧾 Intent Summary:'));
      console.log(chalk.gray(this.indentJson(context.intervention.actionSummary)));
    }

    console.log(chalk.bold.cyan('📋 Arguments:'));

    try {
      const argsJson = JSON.stringify(context.args, null, 2);
      console.log(chalk.gray(this.indentJson(argsJson)));
    } catch {
      console.log(chalk.gray('  [Arguments contain non-serializable data]'));
      console.log(chalk.gray('  ' + String(context.args)));
    }

    console.log('');
    console.log(chalk.dim('─'.repeat(80)));
    console.log('');
  }

  private indentJson(json: string): string {
    return json
      .split('\n')
      .map((line) => '  ' + line)
      .join('\n');
  }

  private async logApprovalDecision(
    context: ExecutionContext,
    decision: 'yes' | 'no' | 'allow' | 'confirm',
    approved: boolean,
    confirmation?: string
  ): Promise<void> {
    try {
      await DecisionLog.append({
        timestamp: new Date().toISOString(),
        module: context.moduleName,
        method: context.methodName,
        args: context.args,
        decision: approved ? 'APPROVED' : 'REJECTED',
        decisionTime: 0,
        reason: 'approval_decision',
        eventType: 'approval_decision',
        approved,
        decisionInput: decision,
        confirmation,
      });
    } catch {
      // best-effort only
    }
  }
}
