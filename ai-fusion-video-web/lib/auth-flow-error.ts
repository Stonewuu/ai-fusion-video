export const AUTH_SESSION_EXPIRED_ERROR_CODE = "AUTH_SESSION_EXPIRED";
export const AUTH_REFRESH_FAILED_ERROR_CODE = "AUTH_REFRESH_FAILED";

export type AuthFlowErrorCode =
  | typeof AUTH_SESSION_EXPIRED_ERROR_CODE
  | typeof AUTH_REFRESH_FAILED_ERROR_CODE;

export class AuthFlowError extends Error {
  readonly code: AuthFlowErrorCode;

  constructor(code: AuthFlowErrorCode, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "AuthFlowError";
    this.code = code;
  }
}
