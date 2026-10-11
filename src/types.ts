import type { ApprovalRequestContext } from './core/ApprovalQueue';

/**
 * Reins type definitions.
 */

/**
 * Security decision types
 * - ALLOW: Execute immediately without prompting
 * - DENY: Block execution and throw error
 * - ASK: Pause and request human decision
 */
export type Decision = 'ALLOW' | 'DENY' | 'ASK';

/**
 * Rule definition for a specific method
 */
export interface SecurityRule {
  action: Decision;
  description?: string; // Optional reasoning for logs and UI
  /**
   * Glob patterns for allowed paths/URLs.
   * If set, the target path/URL MUST match at least one pattern or the call is DENIED.
   * A missing, empty, or non-string target is DENIED when path restrictions are configured.
   * File paths are lexically normalized; symlinks are not resolved. Adapters must
   * enforce any required physical filesystem confinement when executing the operation.
   * Examples: ["/workspace/**", "/tmp/**"], ["https://api.example.com/**"]
   */
  allowPaths?: string[];
  /**
   * Glob patterns for denied paths/URLs.
   * If the target path/URL matches ANY of these, the call is DENIED regardless of allowPaths.
   * A missing, empty, or non-string target is DENIED when path restrictions are configured.
   * Symlink aliases are not resolved to their physical targets before matching.
   * Examples: ["/etc/GLOB", "~/.ssh/GLOB", "GLOB/.env"]  (GLOB = **)
   */
  denyPaths?: string[];
}

/**
 * Complete security policy structure
 */
export interface SecurityPolicy {
  defaultAction: Decision; // Fallback if no rule exists (Paranoia mode)
  modules: {
    [moduleName: string]: {
      [methodName: string]: SecurityRule;
    };
  };
}

export interface InterventionMetadata {
  /** Force human review regardless of normal ALLOW policy (DENY remains DENY). */
  forceAsk?: boolean;
  /** Override the displayed risk description in prompts/logs. */
  overrideDescription?: string;
  /** Extra reason text appended to channel-mode approval instructions. */
  interventionReason?: string;
  /** Escalate from YES/NO to explicit confirmation token flow. */
  requiresExplicitConfirmation?: boolean;
  /** Human-readable summary of the action that needs explicit confirmation. */
  actionSummary?: string;
  /**
   * Confirmation token for TTY prompts. Channel mode ignores this value and
   * generates a fresh token internally for each invocation.
   */
  confirmationToken?: string;
  /**
   * If true, channel instructions tell the agent to capture a screenshot and
   * use vision reasoning before asking for approval.
   */
  recommendScreenshotReview?: boolean;
  /** Current cooldown escalation level (0=normal, 1=heightened, 2=restricted). */
  cooldownLevel?: number;
  /** Destructive-intercept severity for UI and audit context. */
  destructiveSeverity?: 'HIGH' | 'CATASTROPHIC';
  /** Reasons from destructive classifier. */
  destructiveReasons?: string[];
  /** Optional bulk count identified by destructive classifier. */
  destructiveBulkCount?: number;
  /** Optional user-facing target (mailbox/path/host). */
  destructiveTarget?: string;
  /** Require channel approvals to come via reins_respond (fail-secure if unavailable). */
  requiresRespondToolApproval?: boolean;
}

/**
 * Execution context passed to the Arbitrator
 */
export interface ExecutionContext {
  moduleName: string;
  methodName: string;
  args: unknown[];
  rule: SecurityRule;
  /** OpenClaw session key (e.g. "agent:main:whatsapp:dm:+1555…"). Present in daemon/channel mode. */
  sessionKey?: string;
  intervention?: InterventionMetadata;
  /** Called once when a tool is blocked to fire the OOB approval notification.
   *  Returns true if a notification was actually dispatched (channel context found),
   *  false if no channel is available (e.g. TUI sessions). */
  onBlockCallback?: ApprovalCallback;
  /** Trusted invocation binding supplied by Interceptor for channel approval. */
  approvalBinding?: string;
}

/** The fourth argument identifies the exact invocation for a trusted notifier. */
export type ApprovalCallback = (
  sessionKey: string,
  moduleName: string,
  methodName: string,
  request: ApprovalRequestContext
) => Promise<boolean>;
