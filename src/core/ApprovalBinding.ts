/** Strict, bounded JSON binding for channel approvals. Does not invoke user code. */
import { types } from 'node:util';

export const APPROVAL_MAX_DEPTH = 64;
export const APPROVAL_MAX_BYTES = 1024 * 1024;

export interface ApprovalBinding {
  canonical: string;
  snapshot: unknown;
}

export function bindApprovalArguments(value: unknown): ApprovalBinding {
  const ancestors = new Set<object>();
  let bytes = 0;
  const add = (text: string): string => {
    bytes += Buffer.byteLength(text, 'utf8');
    if (bytes > APPROVAL_MAX_BYTES) throw new Error('Channel approval arguments exceed 1 MiB');
    return text;
  };
  const encode = (item: unknown, depth: number): string => {
    if (depth > APPROVAL_MAX_DEPTH) throw new Error('Channel approval arguments exceed depth 64');
    if (item === null) return add('null');
    if (typeof item === 'string' || typeof item === 'boolean') return add(JSON.stringify(item));
    if (typeof item === 'number' && Number.isFinite(item)) {
      return add(Object.is(item, -0) ? '-0' : JSON.stringify(item));
    }
    if (typeof item !== 'object' || types.isProxy(item)) {
      throw new Error('Channel approval arguments must be plain JSON data');
    }
    const proto = Object.getPrototypeOf(item);
    const array = Array.isArray(item);
    if (array ? proto !== Array.prototype : proto !== Object.prototype && proto !== null) {
      throw new Error('Channel approval arguments must use plain objects and arrays');
    }
    if (ancestors.has(item)) throw new Error('Channel approval arguments contain a cycle');
    ancestors.add(item);
    try {
      const keys = Reflect.ownKeys(item);
      if (keys.some((key) => typeof key !== 'string')) {
        throw new Error('Channel approval arguments contain symbol keys');
      }
      const descriptors = Object.getOwnPropertyDescriptors(item);
      if (array) {
        const length = descriptors.length.value as number;
        if (keys.length !== length + 1) throw new Error('Channel approval arrays must be dense');
        const values: string[] = [];
        add('[');
        for (let i = 0; i < length; i += 1) {
          const descriptor = descriptors[String(i)];
          if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
            throw new Error('Channel approval arrays must contain only data elements');
          }
          if (i > 0) add(',');
          values.push(encode(descriptor.value, depth + 1));
        }
        add(']');
        return `[${values.join(',')}]`;
      }
      const fields: string[] = [];
      add('{');
      for (const key of (keys as string[]).sort()) {
        const descriptor = descriptors[key];
        if (!('value' in descriptor) || !descriptor.enumerable) {
          throw new Error('Channel approval objects must contain only enumerable data properties');
        }
        if (fields.length > 0) add(',');
        const encodedKey = add(JSON.stringify(key));
        add(':');
        fields.push(`${encodedKey}:${encode(descriptor.value, depth + 1)}`);
      }
      add('}');
      return `{${fields.join(',')}}`;
    } finally {
      ancestors.delete(item);
    }
  };
  const canonical = encode(value, 0);
  const snapshot: unknown = JSON.parse(canonical);
  const freeze = (item: unknown): void => {
    if (item !== null && typeof item === 'object') {
      for (const child of Object.values(item)) freeze(child);
      Object.freeze(item);
    }
  };
  freeze(snapshot);
  return { canonical, snapshot };
}
