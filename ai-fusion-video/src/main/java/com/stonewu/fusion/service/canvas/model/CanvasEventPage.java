package com.stonewu.fusion.service.canvas.model;

import java.util.List;

public record CanvasEventPage(List<CanvasEvent> events, long latestSequence, boolean hasMore) {
    public CanvasEventPage {
        events = List.copyOf(events);
    }
}
