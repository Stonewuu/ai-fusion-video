import type { SceneItem, DialogueElement } from "@/lib/api/script";

export function parseDialogues(scene: SceneItem): DialogueElement[] {
  if (!scene.dialogues) return [];
  try {
    return typeof scene.dialogues === "string"
      ? JSON.parse(scene.dialogues)
      : scene.dialogues;
  } catch {
    return [];
  }
}

export function parseCharacters(scene: SceneItem): string[] {
  if (!scene.characters) return [];
  try {
    return typeof scene.characters === "string"
      ? JSON.parse(scene.characters)
      : scene.characters;
  } catch {
    return [];
  }
}
