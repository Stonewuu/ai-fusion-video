package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.AssetReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;

/**
 * 集合及父子版本在同一快照中批量读取；单对象详情继续使用版本化缓存。
 */
@Service
public class AssetResourceQueryService {
    private final AssetReadRepository assets;
    private final ResourceQueryService resources;
    private final CanvasAccessPolicy access;
    private final CanvasJson json;
    private final TransactionTemplate reads;

    public AssetResourceQueryService(AssetReadRepository assets, ResourceQueryService resources,
                                     CanvasAccessPolicy access,
                                     CanvasJson json, PlatformTransactionManager manager) {
        this.assets = assets;
        this.resources = resources;
        this.access = access;
        this.json = json;
        reads = new TransactionTemplate(manager);
        reads.setReadOnly(true);
        reads.setIsolationLevel(TransactionDefinition.ISOLATION_REPEATABLE_READ);
    }

    public ObjectNode tree(long assetId, CanvasActor actor) {
        return Objects.requireNonNull(reads.execute(ignored -> {
            ObjectNode asset = resources.get(ASSET, assetId, actor).data();
            ArrayNode children = asset.putArray("items");
            assets.itemDetails(List.of(assetId)).forEach(children::add);
            return asset;
        }));
    }

    public List<ObjectNode> items(long assetId, CanvasActor actor) {
        return Objects.requireNonNull(reads.execute(ignored -> {
            resources.authorizedHeader(ASSET, assetId, actor);
            return assets.itemDetails(List.of(assetId));
        }));
    }

    public List<ObjectNode> list(Long projectId, String type, String keyword, boolean withItems, CanvasActor actor) {
        validateFilters(type, keyword);
        return Objects.requireNonNull(reads.execute(ignored -> {
            List<Long> projects = authorize(projectId, actor);
            List<ObjectNode> rows = assets.details(actor, projects, projectId, type, keyword, null, null);
            if (withItems && !rows.isEmpty()) {
                Map<Long, ArrayNode> children = new LinkedHashMap<>();
                for (ObjectNode row : rows) {
                    children.put(row.path("id").longValue(), row.putArray("items"));
                }
                for (ObjectNode item : assets.itemDetails(List.copyOf(children.keySet()))) {
                    children.get(item.path("assetId").longValue()).add(item);
                }
            }
            return rows;
        }));
    }

    public ObjectNode page(Long projectId, String type, String keyword, int page, int size, CanvasActor actor) {
        if (page < 1 || size < 1 || size > 100) {
            throw AssetFields.invalid("资产分页大小必须为 1 至 100");
        }
        validateFilters(type, keyword);
        return Objects.requireNonNull(reads.execute(ignored -> {
            var projects = authorize(projectId, actor);
            var records = assets.details(actor, projects, projectId, type, keyword, page, size);
            return json.object(
                    Map.of("records", records, "total", assets.count(actor, projects, projectId, type, keyword),
                            "page", page, "size", size, "typeCounts",
                            assets.counts(actor, projects, projectId, keyword)));
        }));
    }

    public ObjectNode summaryPage(Long projectId, String type, String keyword, int page, int size, CanvasActor actor) {
        validatePage(page, size);
        validateFilters(type, keyword);
        return Objects.requireNonNull(reads.execute(ignored -> {
            var projects = authorize(projectId, actor);
            var rows = assets.summaries(actor, projects, projectId, type, keyword, page, size);
            long total = assets.count(actor, projects, projectId, type, keyword);
            ObjectNode result = json.object(Map.of("assets", rows, "total", total));
            pageMetadata(result, page, size, total);
            return result;
        }));
    }

    public ObjectNode itemSummaryPage(long assetId, int page, int size, CanvasActor actor) {
        validatePage(page, size);
        return Objects.requireNonNull(reads.execute(ignored -> {
            resources.authorizedHeader(ASSET, assetId, actor);
            ObjectNode result = assets.summary(assetId);
            var items = assets.itemSummaries(assetId, page, size);
            result.set("items", json.object(Map.of("items", items)).get("items"));
            long total = assets.itemCount(assetId);
            result.put("itemCount", total);
            pageMetadata(result, page, size, total);
            return result;
        }));
    }

    public List<ObjectNode> itemSummaryPages(List<Long> assetIds, int page, int size, CanvasActor actor) {
        validatePage(page, size);
        return Objects.requireNonNull(reads.execute(ignored -> {
            List<Long> projects = authorize(null, actor);
            Map<Long, ObjectNode> summaries = new LinkedHashMap<>();
            for (ObjectNode row : assets.summariesByIds(assetIds, actor, projects)) {
                summaries.put(row.path("id").longValue(), row);
            }
            List<Long> missing = assetIds.stream().filter(id -> !summaries.containsKey(id)).toList();
            if (!missing.isEmpty()) {
                long first = missing.get(0);
                if (!assets.existingIds(missing).contains(first)) {
                    throw new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED, "业务对象已删除或不存在",
                            Map.of("resourceType", ASSET.code(), "resourceId", first));
                }
                throw new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权访问此业务资源", Map.of());
            }
            for (ObjectNode summary : summaries.values()) {
                summary.putArray("items");
                pageMetadata(summary, page, size, summary.path("itemCount").longValue());
            }
            for (ObjectNode item : assets.itemSummariesByAssetIds(assetIds, page, size)) {
                summaries.get(item.path("assetId").longValue()).withArray("items").add(item);
            }
            return assetIds.stream().map(summaries::get).toList();
        }));
    }

    private void validatePage(int page, int size) {
        if (page < 1 || size < 1 || size > 100) {
            throw AssetFields.invalid("页码须大于 0，每页数量为 1 至 100");
        }
    }

    private void pageMetadata(ObjectNode result, int page, int size, long total) {
        boolean hasMore = (long) page * size < total;
        result.put("page", page).put("pageSize", size).put("hasMore", hasMore);
        if (hasMore) {
            result.put("nextPage", page + 1);
        } else {
            result.putNull("nextPage");
        }
    }

    private List<Long> authorize(Long projectId, CanvasActor actor) {
        if (projectId != null) {
            access.requireProject(projectId, actor);
            return List.of(projectId);
        }
        return access.readableProjects(actor);
    }

    private void validateFilters(String type, String keyword) {
        if ((type != null && (type.isBlank() || type.length() > 32)) || (keyword != null && keyword.length() > 200)) {
            throw AssetFields.invalid("资产筛选参数无效");
        }
    }
}
