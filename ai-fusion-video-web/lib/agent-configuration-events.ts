const EVENT = "agent-configuration-changed";
const CHANNEL = "afv-agent-configuration";

/** 只广播用户标识，不跨窗口传递提示词、Skill 内容或凭证。 */
export function notifyAgentConfigurationChanged(userId: number) {
  window.dispatchEvent(new CustomEvent(EVENT, { detail: userId }));
  const channel = new BroadcastChannel(CHANNEL);
  channel.postMessage(userId);
  channel.close();
}

export function subscribeAgentConfiguration(userId: number, refresh: () => void): () => void {
  const channel = new BroadcastChannel(CHANNEL);
  const local = (event: Event) => {
    if ((event as CustomEvent<number>).detail === userId) refresh();
  };
  channel.onmessage = (event: MessageEvent<unknown>) => {
    if (event.data === userId) refresh();
  };
  window.addEventListener(EVENT, local);
  return () => { window.removeEventListener(EVENT, local); channel.close(); };
}
