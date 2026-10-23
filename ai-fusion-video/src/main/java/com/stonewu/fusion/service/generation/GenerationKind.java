package com.stonewu.fusion.service.generation;

public enum GenerationKind {
    IMAGE("afv_image_task", "afv_image_item"), VIDEO("afv_video_task", "afv_video_item");
    private final String taskTable;
    private final String itemTable;

    GenerationKind(String taskTable, String itemTable) {
        this.taskTable = taskTable;
        this.itemTable = itemTable;
    }

    public String taskTable() {
        return taskTable;
    }

    public String itemTable() {
        return itemTable;
    }
}
