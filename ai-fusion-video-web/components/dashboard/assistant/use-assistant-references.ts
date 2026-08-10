"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent,
  type RefObject,
} from "react";
import {
  getAssistantReferenceOptions,
  type AssistantMcpToolReferenceOption,
  type AssistantReferenceOptions,
  type AssistantSkillReferenceOption,
} from "@/lib/api/ai-assistant";
import { projectApi, type Project } from "@/lib/api/project";
import type {
  AssistantCapabilityReference,
  AssistantProjectReference,
  AssistantReferencePickerItem,
  AssistantReferenceTrigger,
} from "./reference-types";

const REFERENCES_CACHE_TTL_MS = 30_000;
let cachedReferenceOptions: AssistantReferenceOptions | null = null;
let cachedReferenceOptionsAt = 0;
let referenceOptionsRequest: Promise<AssistantReferenceOptions> | null = null;
let cachedProjects: Project[] | null = null;
let cachedProjectsAt = 0;
let projectsRequest: Promise<Project[]> | null = null;

function loadReferenceOptions(): Promise<AssistantReferenceOptions> {
  const cached = cachedReferenceOptions;
  if (cached && Date.now() - cachedReferenceOptionsAt < REFERENCES_CACHE_TTL_MS) {
    return Promise.resolve(cached);
  }
  if (!referenceOptionsRequest) {
    referenceOptionsRequest = getAssistantReferenceOptions()
      .then((result) => {
        cachedReferenceOptions = result;
        cachedReferenceOptionsAt = Date.now();
        referenceOptionsRequest = null;
        return result;
      })
      .catch((error: unknown) => {
        referenceOptionsRequest = null;
        throw error;
      });
  }
  return referenceOptionsRequest;
}

function loadProjects(): Promise<Project[]> {
  const cached = cachedProjects;
  if (cached && Date.now() - cachedProjectsAt < REFERENCES_CACHE_TTL_MS) {
    return Promise.resolve(cached);
  }
  if (!projectsRequest) {
    projectsRequest = projectApi.list()
      .then((result) => {
        cachedProjects = result;
        cachedProjectsAt = Date.now();
        projectsRequest = null;
        return result;
      })
      .catch((error: unknown) => {
        projectsRequest = null;
        throw error;
      });
  }
  return projectsRequest;
}

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
  active: boolean;
  text: string;
  selectedConversationId: string | null;
  effectiveProjectId?: number | null;
  inputRef: RefObject<HTMLTextAreaElement | null>;
  updateText: (value: string) => void;
}

interface AssistantReferenceSelectionState {
  conversationId: string | null;
  selectedSkills: AssistantSkillReferenceOption[];
  selectedMcpTools: AssistantMcpToolReferenceOption[];
  projectOverride: AssistantProjectReference | null | undefined;
  picker: AssistantReferenceTrigger | null;
  activeIndex: number;
}

function emptySelection(conversationId: string | null): AssistantReferenceSelectionState {
  return {
    conversationId,
    selectedSkills: [],
    selectedMcpTools: [],
    projectOverride: undefined,
    picker: null,
    activeIndex: 0,
  };
}

export function useAssistantReferences({
  active,
  text,
  selectedConversationId,
  effectiveProjectId,
  inputRef,
  updateText,
}: UseAssistantReferencesOptions) {
  const [referenceOptions, setReferenceOptions] = useState<AssistantReferenceOptions>({
    skills: [],
    mcpTools: [],
  });
  const [projects, setProjects] = useState<Project[]>([]);
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(true);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [capabilitiesError, setCapabilitiesError] = useState<string | null>(null);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [selectionState, setSelectionState] = useState<AssistantReferenceSelectionState>(
    () => emptySelection(selectedConversationId),
  );
  const selection = useMemo(
    () => selectionState.conversationId === selectedConversationId
      ? selectionState
      : emptySelection(selectedConversationId),
    [selectedConversationId, selectionState],
  );
  const updateSelection = useCallback((
    updater: (current: AssistantReferenceSelectionState) => AssistantReferenceSelectionState,
  ) => {
    setSelectionState((current) => updater(
      current.conversationId === selectedConversationId
        ? current
        : emptySelection(selectedConversationId),
    ));
  }, [selectedConversationId]);
  const {
    selectedSkills,
    selectedMcpTools,
    projectOverride,
    picker,
    activeIndex,
  } = selection;

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    void loadReferenceOptions()
      .then((result) => {
        if (cancelled) return;
        setReferenceOptions(result);
        setCapabilitiesError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setCapabilitiesError(error instanceof Error ? error.message : "加载 Skill/MCP 失败");
      })
      .finally(() => {
        if (!cancelled) setCapabilitiesLoading(false);
      });
    void loadProjects()
      .then((result) => {
        if (cancelled) return;
        setProjects(result);
        setProjectsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setProjectsError(error instanceof Error ? error.message : "加载项目失败");
      })
      .finally(() => {
        if (!cancelled) setProjectsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [active]);

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
    clearTransientReferences: () => setSelectionState(emptySelection(selectedConversationId)),
  };
}
