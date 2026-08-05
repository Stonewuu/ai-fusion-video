package com.stonewu.fusion.service.ai.comfyui.client;

/** File descriptor returned by ComfyUI /upload/image. */
public record ComfyUiUploadResult(
        String name,
        String subfolder,
        String type) {

    public String workflowValue() {
        return subfolder == null || subfolder.isBlank() ? name : subfolder + "/" + name;
    }
}
