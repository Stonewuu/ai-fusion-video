package com.stonewu.fusion.service.canvas.model;

public record CanvasPermissions(boolean canView, boolean canEdit, boolean canExecute, boolean canManage) {
    public static CanvasPermissions owner() {
        return new CanvasPermissions(true, true, true, true);
    }

    public static CanvasPermissions viewer() {
        return new CanvasPermissions(true, false, false, false);
    }
}
