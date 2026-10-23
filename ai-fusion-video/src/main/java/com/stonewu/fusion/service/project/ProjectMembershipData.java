package com.stonewu.fusion.service.project;

import com.stonewu.fusion.service.canvas.collaboration.CanvasCollaborationData.Person;
import jakarta.validation.constraints.*;

import java.time.Instant;
import java.util.List;

public final class ProjectMembershipData {
    private ProjectMembershipData() {
    }

    public record Member(Person person, Integer role, boolean protectedOwner, boolean enabled) {
    }

    public record Members(long revision, boolean canManage, boolean teamOwned, List<Member> list, long total) {
    }

    public record Change(@NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
                         @NotNull @PositiveOrZero Long expectedRevision, @Positive long userId, Integer role) {
    }

    public record Receipt(String commandId, long projectId, long revision, long userId, Integer previousRole,
                          Integer currentRole, int removedCanvasGrants, Instant createdAt) {
    }

    public record Audit(Receipt change, Person actor, Person member) {
    }
}
