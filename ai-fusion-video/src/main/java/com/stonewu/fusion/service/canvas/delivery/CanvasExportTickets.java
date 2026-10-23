package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.security.SecureRandom;
import java.time.Duration;
import java.time.Instant;
import java.util.Base64;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasExportTickets {
    private static final SecureRandom RANDOM = new SecureRandom();
    private final CanvasExportService exports;
    private final CanvasExportRepository jobs;
    private final StringRedisTemplate redis;
    private final CanvasJson json;

    public record Ticket(String canvasId, String id, String blobId, CanvasActor actor, Instant expiresAt) {
    }

    public record Issued(String url, Instant expiresAt) {
    }

    public record Authorized(CanvasExportData.Job job, CanvasActor actor) {
    }

    public Issued issue(String canvas, String id, CanvasActor actor) {
        var job = exports.authorized(canvas, id, actor);
        if (!"SUCCEEDED".equals(job.status()) || job.blobId() == null) {
            throw CanvasExportData.conflict("交付文件尚未生成完成");
        }
        byte[] random = new byte[32];
        RANDOM.nextBytes(random);
        String token = Base64.getUrlEncoder().withoutPadding().encodeToString(random);
        var ticket = new Ticket(canvas, id, job.blobId(), actor, Instant.now().plusSeconds(600));
        redis.opsForValue().set(key(token), json.write(ticket), Duration.ofMinutes(10));
        jobs.event(id, actor.userId(), "DOWNLOAD_GRANTED", Map.of("expiresAt", ticket.expiresAt()));
        return new Issued("/api/canvas-export-content/" + id + "?ticket=" + token, ticket.expiresAt());
    }

    public Authorized authorize(String id, String token) {
        if (token == null || !token.matches("[A-Za-z0-9_-]{43}")) {
            throw denied();
        }
        String content = redis.opsForValue().get(key(token));
        if (content == null) {
            throw denied();
        }
        var ticket = json.read(content, Ticket.class);
        if (!ticket.id().equals(id) || !ticket.expiresAt().isAfter(Instant.now())) {
            throw denied();
        }
        var job = exports.authorized(ticket.canvasId(), id, ticket.actor());
        if (!"SUCCEEDED".equals(job.status()) || !ticket.blobId().equals(job.blobId())) {
            throw denied();
        }
        return new Authorized(job, ticket.actor());
    }

    private String key(String token) {
        return "canvas:export-ticket:" + json.hash(token);
    }

    private BusinessException denied() {
        return new BusinessException(403, "交付下载凭据已失效，请重新点击下载");
    }
}
