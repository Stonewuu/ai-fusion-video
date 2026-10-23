package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.service.canvas.model.CanvasPermissions;
import jakarta.validation.constraints.*;

public final class CanvasCollaborationData {
    private CanvasCollaborationData() {
    }

    public record Person(long userId, String username, String name) {
    }

    public record Member(Person person, CanvasPermissions effective, CanvasPermissions granted,
                         String source, boolean protectedOwner) {
    }

    public record Grant(@NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
                        @NotNull @PositiveOrZero Long expectedRevision, @Positive long userId,
                        boolean canEdit, boolean canExecute, boolean canManage, boolean remove) {
    }
}
