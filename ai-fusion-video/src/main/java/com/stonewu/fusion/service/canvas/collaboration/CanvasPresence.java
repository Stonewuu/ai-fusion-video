package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasPermissions;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 存在信息可丢弃；只存账号、会话与选择，不保存正文、不修改共享视口或图版本。
 */
@Component
@RequiredArgsConstructor
public class CanvasPresence {
    private static final Duration TTL = Duration.ofSeconds(45);
    private final StringRedisTemplate redis;
    private final CanvasAccessPolicy access;
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasCollaborationService people;
    private final CanvasRequestValidator validation;
    private final CanvasJson json;

    public record Heartbeat(@NotBlank String sessionId, @NotNull @Size(max = 20) List<String> nodeIds) {
    }

    public record Stored(String sessionId, CanvasActor actor, List<String> nodeIds, long updatedAt) {
    }

    public record Participant(String sessionId, CanvasCollaborationData.Person person, List<String> nodeIds,
                              long updatedAt) {
    }

    public record Snapshot(List<Participant> participants, boolean hasMore, CanvasPermissions permissions) {
    }

    public Snapshot heartbeat(String canvasId, Heartbeat heartbeat, CanvasActor actor) {
        validation.id(canvasId);
        validation.id(heartbeat.sessionId());
        validation.validate(heartbeat);
        heartbeat.nodeIds().forEach(validation::id);
        var document = documents.require(canvasId, false);
        var permissions = access.requireView(document, actor);
        var known = graphs.load(document).nodes().stream().map(node -> node.id()).toList();
        var selection = heartbeat.nodeIds().stream().filter(known::contains).distinct().toList();
        long now = System.currentTimeMillis();
        String member = member(actor, heartbeat.sessionId());
        String index = index(canvasId);
        redis.opsForValue()
                .set(index + ":" + member, json.write(new Stored(heartbeat.sessionId(), actor, selection, now)), TTL);
        redis.opsForZSet().add(index, member, now);
        redis.expire(index, TTL.multipliedBy(2));
        redis.opsForZSet().removeRangeByScore(index, Double.NEGATIVE_INFINITY, now - TTL.toMillis());
        var keys = Objects.requireNonNull(
                redis.opsForZSet().reverseRangeByScore(index, now - TTL.toMillis(), Double.POSITIVE_INFINITY, 0, 101));
        var result = new ArrayList<Participant>();
        for (String key : keys.stream().limit(100).toList()) {
            String value = redis.opsForValue().get(index + ":" + key);
            if (value == null) {
                continue;
            }
            Stored current = json.read(value, Stored.class);
            try {
                access.requireView(document, current.actor());
            } catch (CanvasException denied) {
                if (denied.getCode() != 403) {
                    throw denied;
                }
                redis.opsForZSet().remove(index, key);
                redis.delete(index + ":" + key);
                continue;
            }
            result.add(new Participant(current.sessionId(), people.person(current.actor().userId()), current.nodeIds(),
                    current.updatedAt()));
        }
        return new Snapshot(List.copyOf(result), keys.size() > 100, permissions);
    }

    public void leave(String canvasId, String sessionId, CanvasActor actor) {
        validation.id(canvasId);
        validation.id(sessionId);
        access.requireActorScope(actor);
        // 只能清理本账号这次会话，撤权后也可结束自己的存在信息。
        String index = index(canvasId), member = member(actor, sessionId);
        redis.opsForZSet().remove(index, member);
        redis.delete(index + ":" + member);
    }

    private String member(CanvasActor actor, String sessionId) {
        return actor.userId() + ":" + sessionId;
    }

    private String index(String canvasId) {
        return "canvas:presence:" + canvasId;
    }
}
