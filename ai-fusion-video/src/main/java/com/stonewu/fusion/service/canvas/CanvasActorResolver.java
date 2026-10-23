package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.team.TeamService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class CanvasActorResolver {
    private final TeamService teams;

    public CanvasActor current() {
        long userId = SecurityUtils.requireCurrentUserId();
        return new CanvasActor(userId, teams.getCurrentTeamIdByUser(userId));
    }
}
