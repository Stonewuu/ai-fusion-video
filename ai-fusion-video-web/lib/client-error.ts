import { toast } from "sonner";
import {
  getClientErrorMessage,
  isIgnoredClientError,
} from "@/lib/client-error-policy";

export { getClientErrorMessage } from "@/lib/client-error-policy";

const MAX_ERROR_DESCRIPTION_LENGTH = 240;

function errorFingerprint(message: string): string {
  let hash = 0;
  for (let index = 0; index < message.length; index++) {
    hash = (Math.imul(31, hash) + message.charCodeAt(index)) | 0;
  }
  return (hash >>> 0).toString(36);
}

export function notifyClientError(
  error: unknown,
  context = "页面运行异常"
): string | null {
  const message = getClientErrorMessage(error);
  if (isIgnoredClientError(error)) {
    return null;
  }
  const description = message.length > MAX_ERROR_DESCRIPTION_LENGTH
    ? `${message.slice(0, MAX_ERROR_DESCRIPTION_LENGTH)}…`
    : message;
  toast.error(context, {
    id: `client-error:${errorFingerprint(message)}`,
    description,
    duration: 8000,
  });
  return message;
}
