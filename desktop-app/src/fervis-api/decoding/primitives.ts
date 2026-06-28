export type BoundaryObject = { readonly [key: string]: unknown };

export function decode<T>(
  name: string,
  fn: () => T
): { readonly ok: true; readonly value: T } | { readonly ok: false; readonly error: { readonly message: string } } {
  try {
    return { ok: true, value: fn() };
  } catch (error) {
    const message = error instanceof Error ? error.message : `invalid ${name}`;
    return { ok: false, error: { message } };
  }
}

export function expectObject(raw: unknown, label: string): BoundaryObject {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`${label} must be an object`);
  }
  return raw as BoundaryObject;
}

export function expectArray(raw: unknown, label: string): readonly unknown[] {
  if (!Array.isArray(raw)) {
    throw new Error(`${label} must be an array`);
  }
  return raw;
}

export function expectString(raw: unknown, label: string): string {
  if (typeof raw !== "string") {
    throw new Error(`${label} must be a string`);
  }
  return raw;
}

export function expectNullableString(raw: unknown, label: string): string | null {
  if (raw === null) {
    return null;
  }
  return expectString(raw, label);
}

export function expectNumber(raw: unknown, label: string): number {
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    throw new Error(`${label} must be a finite number`);
  }
  return raw;
}

export function expectBoolean(raw: unknown, label: string): boolean {
  if (typeof raw !== "boolean") {
    throw new Error(`${label} must be a boolean`);
  }
  return raw;
}

export function expectStringArray(raw: unknown, label: string): readonly string[] {
  return expectArray(raw, label).map((item, index) =>
    expectString(item, `${label}[${index}]`)
  );
}
