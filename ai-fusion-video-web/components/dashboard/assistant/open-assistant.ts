const OPEN_ASSISTANT_EVENT = "fusion-video:open-assistant";

export function requestAssistantOpen() {
  window.dispatchEvent(new Event(OPEN_ASSISTANT_EVENT));
}

export function subscribeToAssistantOpenRequest(listener: () => void) {
  window.addEventListener(OPEN_ASSISTANT_EVENT, listener);
  return () => window.removeEventListener(OPEN_ASSISTANT_EVENT, listener);
}
