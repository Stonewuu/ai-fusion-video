import { getConversation, listConversations } from "@/lib/api/ai-assistant";
import { getApiErrorStatus } from "@/lib/api/api-error";
import { ASSISTANT_CATEGORY } from "./assistant-types";

/** Load the first page and explicitly restore an older selected conversation by its owned identity. */
export async function loadInitialAssistantConversations(selectedId: string | null, pageSize: number) {
  const page = await listConversations({ pageNo: 1, pageSize, category: ASSISTANT_CATEGORY });
  const list = page.list.filter((conversation) => !conversation.category || conversation.category === ASSISTANT_CATEGORY);
  if (selectedId && !list.some((conversation) => conversation.conversationId === selectedId)) {
    try {
      const selected = await getConversation(selectedId);
      if (!selected.category || selected.category === ASSISTANT_CATEGORY) list.push(selected);
    } catch (error) {
      // A deleted/inaccessible selection is no longer restorable; transport failures must remain visible and retryable.
      if (getApiErrorStatus(error) !== 404) throw error;
    }
  }
  return { ...page, list };
}
