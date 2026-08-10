<<<<<<< Updated upstream
import { toast } from "sonner";
import { isIgnoredClientError } from "@/lib/client-error-policy";
import { getApiErrorMessage } from "./api-error";

export function toastApiError(error: unknown, context?: string): string {
  const message = getApiErrorMessage(error);
  if (isIgnoredClientError(error)) {
    return message;
  }
  toast.error(message, {
    id: `api-error:${message}`,
    ...(context ? { description: context } : {}),
  });
  return message;
}
=======
import { toast } from "sonner";
import { getApiErrorMessage } from "./api-error";

export function toastApiError(error: unknown, context?: string): string {
  const message = getApiErrorMessage(error);
  toast.error(message, {
    id: `api-error:${message}`,
    ...(context ? { description: context } : {}),
  });
  return message;
}
>>>>>>> Stashed changes
