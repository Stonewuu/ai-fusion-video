export interface SystemConfigs {
  site_base_url: string;
  resource_base_url: string;
  allow_register: boolean;
  mail_smtp_host?: string;
  mail_smtp_port?: string;
  mail_username?: string;
  mail_password?: string;
  mail_ssl?: boolean;
  mail_from?: string;
}

export function normalizePublicUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function isLocalOrPrivateUrl(value: string): boolean {
  if (!value.trim()) return false;
  try {
    const hostname = new URL(value).hostname.toLowerCase();
    if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1") {
      return true;
    }
    if (/^10\./.test(hostname) || /^192\.168\./.test(hostname)) {
      return true;
    }
    const match = hostname.match(/^172\.(\d+)\./);
    return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
  } catch {
    return false;
  }
}
