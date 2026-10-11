/** One channel approval entry per invocation, never a session/method permission. */
import crypto from 'node:crypto';

export interface ApprovalHandle {
  readonly id: string;
}

export interface ApprovalRequestContext {
  readonly handle: ApprovalHandle;
  readonly binding: string;
  readonly summary?: string;
}

export interface ApprovalRequestOptions {
  binding: string;
  requiresExplicitConfirmation?: boolean;
  actionSummary?: string;
}

export interface ApprovalNotification {
  readonly token: string;
  readonly summary?: string;
  readonly binding: string;
  readonly requestId: string;
}

interface ApprovalEntry {
  notification: ApprovalNotification;
  status: 'pending' | 'approved';
  expiresAt: number;
  waiterAttached: boolean;
  waiter?: (approved: boolean) => void;
  timer: ReturnType<typeof setTimeout>;
  closed: Promise<void>;
  close: () => void;
}

const DEFAULT_TTL_MS = 120_000;

export class ApprovalQueue {
  private entries = new Map<ApprovalHandle, ApprovalEntry>();
  private tokens = new Map<string, ApprovalHandle>();

  constructor(private readonly ttl: number = DEFAULT_TTL_MS) {
    if (!Number.isFinite(ttl) || ttl <= 0 || ttl > 2_147_483_647) {
      throw new Error('Approval TTL must be a positive, bounded number of milliseconds');
    }
  }

  request(
    sessionKey: string,
    moduleName: string,
    methodName: string,
    options: ApprovalRequestOptions
  ): ApprovalRequestContext | undefined {
    // An old unbound request cannot create a runtime approval.
    if (
      !sessionKey ||
      !moduleName ||
      !methodName ||
      typeof options?.binding !== 'string' ||
      !options.binding
    ) {
      return undefined;
    }
    const handle = Object.freeze({ id: crypto.randomUUID() });
    let token: string;
    do {
      token = `CONFIRM-${crypto.randomBytes(16).toString('hex').toUpperCase()}`;
    } while (this.tokens.has(token));
    const notification = Object.freeze({
      token,
      summary: options.actionSummary,
      binding: options.binding,
      requestId: handle.id,
    });
    let close!: () => void;
    const closed = new Promise<void>((resolve) => {
      close = resolve;
    });
    const entry: ApprovalEntry = {
      notification,
      status: 'pending',
      expiresAt: Date.now() + this.ttl,
      waiterAttached: false,
      timer: setTimeout(() => this.finish(handle, false), this.ttl),
      closed,
      close,
    };
    this.entries.set(handle, entry);
    this.tokens.set(token, handle);
    return Object.freeze({ handle, binding: options.binding, summary: options.actionSummary });
  }

  private live(handle: ApprovalHandle): ApprovalEntry | undefined {
    const entry = this.entries.get(handle);
    if (entry && Date.now() >= entry.expiresAt) {
      this.finish(handle, false);
      return undefined;
    }
    return entry;
  }

  private finish(handle: ApprovalHandle, approved: boolean): boolean {
    const entry = this.entries.get(handle);
    if (!entry) return false;
    this.entries.delete(handle);
    this.tokens.delete(entry.notification.token);
    clearTimeout(entry.timer);
    entry.close();
    entry.waiter?.(approved);
    return true;
  }

  cancel(handle: ApprovalHandle): boolean {
    return this.finish(handle, false);
  }

  getNotificationInfo(handle: ApprovalHandle): ApprovalNotification | undefined {
    const entry = this.live(handle);
    return entry?.status === 'pending' ? entry.notification : undefined;
  }

  resolveByToken(token: string, decision: 'approve' | 'deny'): boolean {
    if (typeof token !== 'string' || (decision !== 'approve' && decision !== 'deny')) return false;
    const handle = this.tokens.get(token.trim().toUpperCase());
    if (!handle) return false;
    const entry = this.live(handle);
    if (!entry || entry.status !== 'pending') return false;
    if (decision === 'deny') this.finish(handle, false);
    else {
      entry.status = 'approved';
      if (entry.waiterAttached) this.finish(handle, true);
    }
    return true;
  }

  /** Notification failures and hanging callbacks are bounded by the original TTL. */
  async notify(handle: ApprovalHandle, callback: () => Promise<boolean>): Promise<boolean> {
    const entry = this.live(handle);
    if (!entry) return false;
    const notified = await Promise.race([
      Promise.resolve()
        .then(callback)
        .then(
          (result) => result === true,
          () => false
        ),
      entry.closed.then(() => false),
    ]);
    if (!notified || !this.live(handle)) {
      this.cancel(handle);
      return false;
    }
    return true;
  }

  /** A second waiter never joins an invocation or consumes its approval. */
  waitForApproval(handle: ApprovalHandle, timeoutMs?: number): Promise<boolean> {
    const entry = this.live(handle);
    if (!entry || entry.waiterAttached) return Promise.resolve(false);
    entry.waiterAttached = true;
    if (timeoutMs !== undefined) {
      if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
        this.cancel(handle);
        return Promise.resolve(false);
      }
      const now = Date.now();
      const expiresAt = Math.min(entry.expiresAt, now + timeoutMs);
      if (expiresAt < entry.expiresAt) {
        entry.expiresAt = expiresAt;
        clearTimeout(entry.timer);
        entry.timer = setTimeout(() => this.finish(handle, false), Math.max(0, expiresAt - now));
      }
    }
    if (entry.status === 'approved') {
      this.finish(handle, true);
      return Promise.resolve(true);
    }
    return new Promise<boolean>((resolve) => {
      entry.waiter = resolve;
    });
  }
}

export const approvalQueue = new ApprovalQueue();
