export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = body?.error?.message || body?.detail || msg;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(msg, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export async function get<T = any>(path: string): Promise<T> {
  return handle<T>(await fetch(API + path, { cache: "no-store" }));
}

export async function post<T = any>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export async function upload<T = any>(path: string, form: FormData): Promise<T> {
  return handle<T>(await fetch(API + path, { method: "POST", body: form }));
}

export async function del(path: string): Promise<void> {
  return handle<void>(await fetch(API + path, { method: "DELETE" }));
}

export function download(filename: string, content: string, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const fmt = (n: number | null | undefined, digits = 0) =>
  n === null || n === undefined || Number.isNaN(n)
    ? "—"
    : n.toLocaleString(undefined, { maximumFractionDigits: digits });

export const errMsg = (e: unknown) =>
  e instanceof Error ? e.message : "Request failed. Is the backend running on port 8000?";
