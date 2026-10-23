package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.ResourceCommandRepository;
import com.stonewu.fusion.service.canvas.CanvasSnapshotCache;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@Service
public class ResourceEventProjectionService {
    private final ResourceCommandRepository events;
    private final CanvasDocumentRepository canvases;
    private final CanvasSnapshotCache snapshots;
    private final TransactionTemplate writes;

    public ResourceEventProjectionService(ResourceCommandRepository events, CanvasDocumentRepository canvases,
                                          CanvasSnapshotCache snapshots, PlatformTransactionManager manager) {
        this.events = events;
        this.canvases = canvases;
        this.snapshots = snapshots;
        this.writes = new TransactionTemplate(manager);
    }

    private record Projection(boolean published, List<CanvasDocument> changed) {
    }

    public boolean publishNext() {
        Projection result = Objects.requireNonNull(writes.execute(ignored -> {
            var next = events.nextEvent();
            if (next.isEmpty()) {
                return new Projection(false, List.of());
            }
            var event = next.get();
            List<CanvasDocument> changed = new ArrayList<>();
            for (String id : events.targetCanvases(event.change())) {
                CanvasDocument canvas = canvases.require(id, true);
                events.project(event, id, canvas.nextEventSequence());
                canvases.advance(canvas);
                changed.add(canvas);
            }
            events.markPublished(event.id());
            return new Projection(true, List.copyOf(changed));
        }));
        result.changed().forEach(snapshots::evict);
        return result.published();
    }
}
