package com.stonewu.fusion.service.ai.agentscope.workspace;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import io.agentscope.harness.agent.filesystem.remote.store.BaseStore;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class AgentWorkspaceBaseStore implements BaseStore {

    private static final TypeReference<Map<String, Object>> VALUE_TYPE = new TypeReference<>() {
    };

    private final AgentWorkspaceEntryMapper entryMapper;
    private final AgentWorkspaceConfigService configService;
    private final AgentWorkspacePayloadService payloadService;
    private final ObjectMapper objectMapper;

    @Override
    public StoreItem get(List<String> namespace, String key) {
        AgentWorkspaceEntry entry = find(namespace, key);
        return entry == null ? null : toStoreItem(entry);
    }

    @Override
    @Transactional
    public synchronized void put(List<String> namespace, String key, Map<String, Object> value) {
        write(namespace, key, value, null);
    }

    @Override
    @Transactional
    public synchronized boolean putIfVersion(
            List<String> namespace,
            String key,
            Map<String, Object> value,
            long expectedVersion) {
        return write(namespace, key, value, expectedVersion);
    }

    @Override
    public List<StoreItem> search(List<String> namespace, int limit, int offset) {
        int safeLimit = Math.max(0, Math.min(limit, 1000));
        int safeOffset = Math.max(0, offset);
        if (safeLimit == 0) {
            return List.of();
        }
        String namespaceKey = namespaceKey(namespace);
        String namespaceHash = AgentWorkspacePayloadService.sha256(namespaceKey);
        return entryMapper.selectList(new LambdaQueryWrapper<AgentWorkspaceEntry>()
                        .eq(AgentWorkspaceEntry::getNamespaceHash, namespaceHash)
                        .orderByAsc(AgentWorkspaceEntry::getItemKey)
                        .last("LIMIT " + safeLimit + " OFFSET " + safeOffset))
                .stream()
                .filter(entry -> namespaceKey.equals(entry.getNamespaceKey()))
                .map(this::toStoreItem)
                .toList();
    }

    @Override
    @Transactional
    public synchronized void delete(List<String> namespace, String key) {
        configService.lockWritableLocation();
        String namespaceKey = namespaceKey(namespace);
        AgentWorkspaceEntry entry = entryMapper.selectForUpdate(
                AgentWorkspacePayloadService.sha256(namespaceKey),
                AgentWorkspacePayloadService.sha256(requireKey(key)));
        if (entry != null
                && namespaceKey.equals(entry.getNamespaceKey())
                && key.equals(entry.getItemKey())) {
            entryMapper.physicalDeleteById(entry.getId());
            payloadService.delete(stored(entry));
        }
    }

    public AgentWorkspaceEntry find(List<String> namespace, String key) {
        String namespaceKey = namespaceKey(namespace);
        String safeKey = requireKey(key);
        AgentWorkspaceEntry entry = entryMapper.selectOne(new LambdaQueryWrapper<AgentWorkspaceEntry>()
                .eq(AgentWorkspaceEntry::getNamespaceHash,
                        AgentWorkspacePayloadService.sha256(namespaceKey))
                .eq(AgentWorkspaceEntry::getItemHash,
                        AgentWorkspacePayloadService.sha256(safeKey))
                .last("LIMIT 1"));
        if (entry == null
                || !namespaceKey.equals(entry.getNamespaceKey())
                || !safeKey.equals(entry.getItemKey())) {
            return null;
        }
        return entry;
    }

    private boolean write(
            List<String> namespace,
            String key,
            Map<String, Object> value,
            Long expectedVersion) {
        Objects.requireNonNull(value, "value must not be null");
        String namespaceKey = namespaceKey(namespace);
        String safeKey = requireKey(key);
        String namespaceHash = AgentWorkspacePayloadService.sha256(namespaceKey);
        String itemHash = AgentWorkspacePayloadService.sha256(safeKey);
        AgentWorkspaceEntry current = entryMapper.selectForUpdate(namespaceHash, itemHash);
        if (current != null && (!namespaceKey.equals(current.getNamespaceKey())
                || !safeKey.equals(current.getItemKey()))) {
            throw new IllegalStateException("SHA-256 collision in AgentScope workspace store");
        }
        if (expectedVersion != null) {
            if (expectedVersion == 0 && current != null) {
                return false;
            }
            if (expectedVersion > 0
                    && (current == null || current.getVersion() == null
                    || !Objects.equals(current.getVersion(), expectedVersion))) {
                return false;
            }
        }

        String payload;
        try {
            payload = objectMapper.writeValueAsString(value);
        } catch (Exception failure) {
            throw new IllegalArgumentException("Cannot serialize AgentScope workspace value", failure);
        }
        AgentWorkspaceStoredPayload stored = payloadService.write(
                payload, configService.lockWritableLocation());
        if (current == null) {
            AgentWorkspaceEntry created = AgentWorkspaceEntry.builder()
                    .namespaceHash(namespaceHash)
                    .namespaceKey(namespaceKey)
                    .itemHash(itemHash)
                    .itemKey(safeKey)
                    .version(1L)
                    .build();
            applyStored(created, stored);
            try {
                entryMapper.insert(created);
                return true;
            } catch (DuplicateKeyException duplicate) {
                payloadService.delete(stored);
                return false;
            }
        }
        AgentWorkspaceStoredPayload previous = stored(current);
        current.setVersion(current.getVersion() + 1);
        applyStored(current, stored);
        entryMapper.updateById(current);
        payloadService.delete(previous);
        return true;
    }

    private StoreItem toStoreItem(AgentWorkspaceEntry entry) {
        try {
            Map<String, Object> value = objectMapper.readValue(payloadService.read(entry), VALUE_TYPE);
            return new StoreItem(entry.getItemKey(), value, entry.getVersion());
        } catch (Exception failure) {
            throw new IllegalStateException(
                    "Cannot read AgentScope workspace entry " + entry.getItemKey(), failure);
        }
    }

    private void applyStored(AgentWorkspaceEntry entry, AgentWorkspaceStoredPayload stored) {
        entry.setBackendType(stored.backendType());
        entry.setStorageConfigId(stored.storageConfigId());
        entry.setLocalPath(stored.localPath());
        entry.setContentRef(stored.contentRef());
        entry.setPayload(stored.databasePayload());
        entry.setContentSha256(stored.sha256());
        entry.setContentSize(stored.size());
    }

    private AgentWorkspaceStoredPayload stored(AgentWorkspaceEntry entry) {
        return new AgentWorkspaceStoredPayload(
                entry.getBackendType(),
                entry.getStorageConfigId(),
                entry.getLocalPath(),
                entry.getContentRef(),
                entry.getPayload(),
                entry.getContentSha256(),
                entry.getContentSize() == null ? 0 : entry.getContentSize());
    }

    private String namespaceKey(List<String> namespace) {
        if (namespace == null || namespace.isEmpty()
                || namespace.stream().anyMatch(value -> value == null || value.isBlank())) {
            throw new IllegalArgumentException("namespace must contain non-blank segments");
        }
        return String.join("/", namespace);
    }

    private String requireKey(String key) {
        if (key == null || key.isBlank()) {
            throw new IllegalArgumentException("workspace key must not be blank");
        }
        return key;
    }
}
