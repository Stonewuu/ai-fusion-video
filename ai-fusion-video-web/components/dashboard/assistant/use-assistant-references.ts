"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent,
  type RefObject,
} from "react";
import type {
  AssistantAgentType,
  AssistantMcpToolReferenceOption,
  AssistantReferenceOptions,
  AssistantSkillReferenceOption,
} from "@/lib/api/ai-assistant";
import type { Project } from "@/lib/api/project";
import { invalidateAssistantReferences, loadAssistantProjects, loadAssistantReferenceOptions } from "./catalog-cache";
import { subscribeAgentConfiguration } from "@/lib/agent-configuration-events";
import type {
  AssistantCapabilityReference,
  AssistantProjectReference,
  AssistantReferencePickerItem,
  AssistantReferenceTrigger,
} from "./reference-types";

const EMPTY_REFERENCES: AssistantReferenceOptions = { skills: [], mcpTools: [] };
const EMPTY_PROJECTS: Project[] = [];
interface CatalogState<T> { scopeKey: string; value: T; error: string | null }

function detectReferenceTrigger(value: string, cursor: number): AssistantReferenceTrigger | null {
  const prefix = value.slice(0, cursor);
  const match = prefix.match(/(^|\s)([/@])([^\s/@]*)$/u);
  if (!match || match.index === undefined) return null;
  const leadingLength = match[1].length;
  return {
    mode: match[2] === "@" ? "project" : "capability",
    query: match[3],
    start: match.index + leadingLength,
    end: cursor,
  };
}

function normalizeSearch(value: string) {
  return value.trim().toLocaleLowerCase();
}

function fuzzyScore(candidate: string, query: string) {
  if (!query) return 0;
  const normalizedCandidate = normalizeSearch(candidate);
  const normalizedQuery = normalizeSearch(query);
  const direct = normalizedCandidate.indexOf(normalizedQuery);
  if (direct >= 0) return direct;
  let queryIndex = 0;
  let distance = 0;
  for (let index = 0; index < normalizedCandidate.length && queryIndex < normalizedQuery.length; index += 1) {
    if (normalizedCandidate[index] === normalizedQuery[queryIndex]) {
      distance += index;
      queryIndex += 1;
    }
  }
  return queryIndex === normalizedQuery.length ? 100 + distance : Number.POSITIVE_INFINITY;
}

function capabilitySearchText(item: AssistantCapabilityReference) {
  if (item.kind === "skill") {
    return `${item.value.displayName} ${item.value.name} ${item.value.id} ${item.value.source} ${item.value.description}`;
  }
  return `${item.value.toolName} ${item.value.serverName} ${item.value.description}`;
}

interface UseAssistantReferencesOptions {
  userId: number | null;
  agentType: AssistantAgentType;
  active: boolean;
  text: string;
  selectedConversationId: string | null;
  effectiveProjectId?: number | null;
  inputRef: RefObject<HTMLTextAreaElement | null>;
  updateText: (value: string) => void;
}

interface AssistantReferenceSelectionState {
  scopeKey: string;
  selectedSkills: AssistantSkillReferenceOption[];
  selectedMcpTools: AssistantMcpToolReferenceOption[];
  projectOverride: AssistantProjectReference | null | undefined;
  picker: AssistantReferenceTrigger | null;
  activeIndex: number;
}

function emptySelection(scopeKey: string): AssistantReferenceSelectionState {
  return {
    scopeKey,
    selectedSkills: [],
    selectedMcpTools: [],
    projectOverride: undefined,
    picker: null,
    activeIndex: 0,
  };
}

export function useAssistantReferences({
  userId,
  agentType,
  active,
  text,
  selectedConversationId,
  effectiveProjectId,
  inputRef,
  updateText,
}: UseAssistantReferencesOptions) {
  const referenceScope = `${userId}:${agentType}`;
  const [configurationRevision, setConfigurationRevision] = useState(0);
  useEffect(() => {
    if (!userId) return;
    return subscribeAgentConfiguration(userId, () => {
      invalidateAssistantReferences(userId);
      setConfigurationRevision((value) => value + 1);
    });
  }, [userId]);
  const projectScope = String(userId);
  const selectionScope = `${referenceScope}:${selectedConversationId ?? "new"}`;
  const [referenceState, setReferenceState] = useState<CatalogState<AssistantReferenceOptions> | null>(null);
  const [projectState, setProjectState] = useState<CatalogState<Project[]> | null>(null);
  const currentReferences = referenceState?.scopeKey === referenceScope ? referenceState : null;
  const currentProjects = projectState?.scopeKey === projectScope ? projectState : null;
  const referenceOptions = currentReferences?.value ?? EMPTY_REFERENCES;
  const projects = currentProjects?.value ?? EMPTY_PROJECTS;
  const capabilitiesLoading = !!userId && !currentReferences;
  const projectsLoading = !!userId && !currentProjects;
  const capabilitiesError = currentReferences?.error ?? null;
  const projectsError = currentProjects?.error ?? null;
  const [selectionState, setSelectionState] = useState<AssistantReferenceSelectionState>(
    () => emptySelection(selectionScope),
  );
  const selection = useMemo(
    () => selectionState.scopeKey === selectionScope
      ? selectionState
      : emptySelection(selectionScope),
    [selectionScope, selectionState],
  );
  const updateSelection = useCallback((
    updater: (current: AssistantReferenceSelectionState) => AssistantReferenceSelectionState,
  ) => {
    setSelectionState((current) => updater(
      current.scopeKey === selectionScope
        ? current
        : emptySelection(selectionScope),
    ));
  }, [selectionScope]);
  const {
    selectedSkills,
    selectedMcpTools,
    projectOverride,
    picker,
    activeIndex,
  } = selection;

  useEffect(() => {
    if (!active || !userId) return;
    let cancelled = false;
    void loadAssistantReferenceOptions(userId, agentType)
      .then((result) => {
        if (cancelled) return;
        setReferenceState({ scopeKey: referenceScope, value: result, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setReferenceState({ scopeKey: referenceScope, value: EMPTY_REFERENCES,
          error: error instanceof Error ? error.message : "加载 Skill/MCP 失败" });
      });
    void loadAssistantProjects(userId)
      .then((result) => {
        if (cancelled) return;
        setProjectState({ scopeKey: projectScope, value: result, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setProjectState({ scopeKey: projectScope, value: EMPTY_PROJECTS,
          error: error instanceof Error ? error.message : "加载项目失败" });
      });
    return () => {
      cancelled = true;
    };
  }, [active, agentType, projectScope, referenceScope, userId, configurationRevision]);

  const inheritedProject = useMemo<AssistantProjectReference | null>(() => {
    if (!effectiveProjectId) return null;
    const project = projects.find((item) => item.id === effectiveProjectId);
    return {
      id: effectiveProjectId,
      name: project?.name || `项目 #${effectiveProjectId}`,
      description: project?.description,
    };
  }, [effectiveProjectId, projects]);
  const selectedProject = projectOverride === undefined ? inheritedProject : projectOverride;

  const capabilityItems = useMemo<AssistantCapabilityReference[]>(() => [
    ...referenceOptions.skills.map((value) => ({ kind: "skill" as const, value })),
    ...referenceOptions.mcpTools.map((value) => ({ kind: "mcp" as const, value })),
  ], [referenceOptions]);

  const pickerItems = useMemo<AssistantReferencePickerItem[]>(() => {
    if (!picker) return [];
    const candidates: AssistantReferencePickerItem[] = picker.mode === "project"
      ? projects.map((project) => ({
          kind: "project" as const,
          value: { id: project.id, name: project.name, description: project.description },
        }))
      : capabilityItems;
    const query = picker.query;
    return candidates
      .map((item) => {
        const searchText = item.kind === "project"
          ? `${item.value.name} ${item.value.id} ${item.value.description || ""}`
          : capabilitySearchText(item);
        return { item, score: fuzzyScore(searchText, query) };
      })
      .filter((entry) => Number.isFinite(entry.score))
      .sort((left, right) => left.score - right.score)
      .slice(0, 12)
      .map((entry) => entry.item);
  }, [capabilityItems, picker, projects]);

  const selectedKeys = useMemo(() => new Set([
    ...(selectedProject ? [`project:${selectedProject.id}`] : []),
    ...selectedSkills.map((skill) => `skill:${skill.id}`),
    ...selectedMcpTools.map((tool) => `mcp:${tool.serverName}:${tool.toolName}`),
  ]), [selectedMcpTools, selectedProject, selectedSkills]);

  const replaceTrigger = useCallback(() => {
    if (!picker) return;
    const next = text.slice(0, picker.start) + text.slice(picker.end);
    updateText(next);
    const cursor = picker.start;
    requestAnimationFrame(() => {
      inputRef.current?.focus({ preventScroll: true });
      inputRef.current?.setSelectionRange(cursor, cursor);
    });
    updateSelection((current) => ({ ...current, picker: null }));
  }, [inputRef, picker, text, updateSelection, updateText]);

  const selectPickerItem = useCallback((item: AssistantReferencePickerItem) => {
    updateSelection((current) => {
      if (item.kind === "project") {
        return { ...current, projectOverride: item.value };
      }
      if (item.kind === "skill") {
        return current.selectedSkills.some((skill) => skill.id === item.value.id)
          ? current
          : { ...current, selectedSkills: [...current.selectedSkills, item.value] };
      }
      return current.selectedMcpTools.some((tool) =>
        tool.serverName === item.value.serverName && tool.toolName === item.value.toolName)
        ? current
        : { ...current, selectedMcpTools: [...current.selectedMcpTools, item.value] };
    });
    replaceTrigger();
  }, [replaceTrigger, updateSelection]);

  const updateTextWithTrigger = useCallback((value: string, cursor: number) => {
    updateText(value);
    const nextPicker = detectReferenceTrigger(value, cursor);
    updateSelection((current) => ({
      ...current,
      picker: nextPicker,
      activeIndex: current.picker?.mode === nextPicker?.mode
        && current.picker?.query === nextPicker?.query
        ? current.activeIndex
        : 0,
    }));
  }, [updateSelection, updateText]);

  const setActiveIndex = useCallback((index: number) => {
    updateSelection((current) => ({ ...current, activeIndex: index }));
  }, [updateSelection]);

  const closePicker = useCallback(() => {
    updateSelection((current) => current.picker
      ? { ...current, picker: null }
      : current);
  }, [updateSelection]);

  const handlePickerKeyDown = useCallback((event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!picker) return false;
    if (event.key === "Escape") {
      event.preventDefault();
      closePicker();
      return true;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!pickerItems.length) return true;
      setActiveIndex(event.key === "ArrowDown"
        ? (activeIndex + 1) % pickerItems.length
        : (activeIndex - 1 + pickerItems.length) % pickerItems.length);
      return true;
    }
    if ((event.key === "Enter" || event.key === "Tab") && pickerItems[activeIndex]) {
      event.preventDefault();
      selectPickerItem(pickerItems[activeIndex]);
      return true;
    }
    return false;
  }, [activeIndex, closePicker, picker, pickerItems, selectPickerItem, setActiveIndex]);

  return {
    selectedProject,
    selectedSkills,
    selectedMcpTools,
    picker,
    pickerItems,
    activeIndex,
    selectedKeys,
    pickerLoading: picker?.mode === "project" ? projectsLoading : capabilitiesLoading,
    pickerError: picker?.mode === "project" ? projectsError : capabilitiesError,
    setActiveIndex,
    selectPickerItem,
    updateTextWithTrigger,
    handlePickerKeyDown,
    closePicker,
    removeProject: () => updateSelection((current) => ({ ...current, projectOverride: null })),
    removeSkill: (id: string) => updateSelection((current) => ({
      ...current,
      selectedSkills: current.selectedSkills.filter((skill) => skill.id !== id),
    })),
    removeMcpTool: (serverName: string, toolName: string) => updateSelection((current) => ({
      ...current,
      selectedMcpTools: current.selectedMcpTools.filter((tool) =>
        tool.serverName !== serverName || tool.toolName !== toolName),
    })),
    clearTransientReferences: () => setSelectionState(emptySelection(selectionScope)),
  };
}
