package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.AssetReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;

/**
 * 集合查询与详情版本在同一快照读取，内容缓存由 ResourceQueryService 的版本化缓存承担。
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
            asset.set("items", json.object(Map.of("items", items(assetId, actor))).get("items"));
            return asset;
        }));
    }

    public List<ObjectNode> items(long assetId, CanvasActor actor) {
        return Objects.requireNonNull(reads.execute(ignored -> {
            resources.get(ASSET, assetId, actor);
            return assets.items(assetId).stream().map(id -> resources.get(ASSET_ITEM, id, actor).data()).toList();
        }));
    }

    public List<ObjectNode> list(Long projectId, String type, String keyword, boolean withItems, CanvasActor actor) {
        validateFilters(type, keyword);
        return Objects.requireNonNull(reads.execute(ignored -> {
            List<Long> projects = authorize(projectId, actor);
            return assets.ids(actor, projects, projectId, type, keyword, null, null).stream()
                    .map(id -> withItems ? tree(id, actor) : resources.get(ASSET, id, actor).data()).toList();
        }));
    }

    public ObjectNode page(Long projectId, String type, String keyword, int page, int size, CanvasActor actor) {
        if (page < 1 || size < 1 || size > 100) {
            throw AssetFields.invalid("资产分页大小必须为 1 至 100");
        }
        validateFilters(type, keyword);
        return Objects.requireNonNull(reads.execute(ignored -> {
            var projects = authorize(projectId, actor);
            var records = assets.ids(actor, projects, projectId, type, keyword, page, size).stream()
                    .map(id -> resources.get(ASSET, id, actor).data()).toList();
            return json.object(
                    Map.of("records", records, "total", assets.count(actor, projects, projectId, type, keyword),
                            "page", page, "size", size, "typeCounts",
                            assets.counts(actor, projects, projectId, keyword)));
        }));
    }

    private List<Long> authorize(Long projectId, CanvasActor actor) {
        access.requireActorScope(actor);
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
