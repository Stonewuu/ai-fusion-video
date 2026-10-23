package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.canvas.model.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
public class CanvasQueryService {
    private final CanvasDocumentRepository documents;
    private final CanvasJournalRepository journal;
    private final CanvasAccessPolicy access;
    private final CanvasSnapshotCache snapshots;
    private final CanvasJson json;
    private final CanvasRequestValidator requests;
    private final TransactionTemplate reads;

    public CanvasQueryService(CanvasDocumentRepository documents, CanvasJournalRepository journal,
                              CanvasAccessPolicy access, CanvasSnapshotCache snapshots, CanvasJson json,
                              CanvasRequestValidator requests, PlatformTransactionManager manager) {
        this.documents = documents;
        this.journal = journal;
        this.access = access;
        this.snapshots = snapshots;
        this.json = json;
        this.requests = requests;
        this.reads = new TransactionTemplate(manager);
        reads.setReadOnly(true);
        reads.setIsolationLevel(TransactionDefinition.ISOLATION_REPEATABLE_READ);
    }

    public CanvasSnapshot snapshot(String canvasId, CanvasActor actor) {
        requests.id(canvasId);
        return Objects.requireNonNull(reads.execute(ignored -> {
            CanvasDocument document = documents.require(canvasId, false);
            CanvasPermissions permissions = access.requireView(document, actor);
            CanvasGraph graph = json.read(snapshots.graph(document), CanvasGraph.class);
            return new CanvasSnapshot(document, graph, document.nextEventSequence() - 1, permissions);
        }));
    }

    public PageResult<CanvasListItem> list(CanvasActor actor, Long projectId, String search, boolean archived,
                                           int pageNo, int pageSize) {
        if (pageNo < 1 || pageSize < 1 || pageSize > 100) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "分页参数无效，每页最多 100 项");
        }
        String keyword = search.trim();
        if (keyword.length() > 200) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "搜索内容不能超过 200 个字符");
        }
        return Objects.requireNonNull(reads.execute(ignored -> {
            if (projectId != null) {
                access.requireProject(projectId, actor);
            }
            return documents.list(actor.userId(), access.readableProjects(actor), projectId, keyword, archived, pageNo,
                    pageSize);
        }));
    }

    public CanvasEventPage events(String canvasId, CanvasActor actor, long afterSequence, int limit) {
        requests.id(canvasId);
        if (afterSequence < 0 || limit < 1 || limit > 500) {
            throw new CanvasException(CanvasErrorCode.INVALID_CURSOR, "事件游标或读取数量无效");
        }
        return Objects.requireNonNull(reads.execute(ignored -> {
            CanvasDocument document = documents.require(canvasId, false);
            access.requireView(document, actor);
            long latest = document.nextEventSequence() - 1;
            if (afterSequence > latest) {
                throw new CanvasException(409, CanvasErrorCode.INVALID_CURSOR, "事件游标超过画布最新位置",
                        Map.of("latestSequence", latest));
            }
            Long first = journal.firstSequence(canvasId);
            if (afterSequence < latest && (first == null || afterSequence < first - 1)) {
                throw expired(latest);
            }
            List<CanvasEvent> events = journal.events(canvasId, afterSequence, limit);
            long next = afterSequence;
            for (CanvasEvent event : events) {
                if (event.sequence() != ++next) {
                    throw expired(latest);
                }
            }
            if (events.isEmpty() && afterSequence < latest) {
                throw expired(latest);
            }
            return new CanvasEventPage(events, latest, next < latest);
        }));
    }

    private CanvasException expired(long latest) {
        return new CanvasException(409, CanvasErrorCode.SNAPSHOT_REQUIRED, "事件记录已不连续，请重新读取画布快照",
                Map.of("latestSequence", latest));
    }
}
