package com.stonewu.fusion.service.media;

import com.stonewu.fusion.mapper.media.MediaAccessTicketMapper;
import com.stonewu.fusion.entity.media.MediaAccessTicketEntity;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Claim;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.RemoteGenerationJob;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Objects;

/**
 * 不缓存授权；浏览器、生成尝试和 Agent 工作者各有明确的凭据生命周期。
 */
@Component
@RequiredArgsConstructor
public class MediaAccessTickets {
    private static final SecureRandom RANDOM = new SecureRandom();
    private final MediaAccessTicketMapper tickets;
    private final ExecutionCoordinationMapper clock;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final GenerationJobRepository generationJobs;

    public record Issued(String resourceId, String url, Instant expiresAt, boolean download) {
    }

    public record Authorized(MediaResource resource, String blobId, CanvasActor actor, boolean download) {
    }

    @Transactional
    public Issued browser(String id, CanvasActor actor, boolean download) {
        var expiry = clock.now().plusMinutes(5);
        return issue(id, actor, download, new Binding("BROWSER", expiry, null, null, null, null, null, null));
    }

    @Transactional
    public Issued generation(String id, CanvasActor actor, RemoteGenerationJob.Claim claim) {
        var job = generationJobs.leased(claim, false);
        var submission = generationJobs.submission(job.submissionId());
        if (!submission.context().actor().equals(actor) || !job.phase()
                .equals("PREPARING") || job.activeAttemptId() == null) {
            throw denied();
        }
        var media = resources.get(id, false).resource();
        if (media.projectId() != null && !Objects.equals(media.projectId(), submission.projectId())) {
            throw new MediaException(403, "MEDIA_INPUT_SCOPE_MISMATCH", "输入素材来自其他项目，请先显式导入");
        }
        return issue(id, actor, false,
                new Binding("GENERATION", job.deadline(), job.id(), job.activeAttemptId(), null, null, null, null));
    }

    @Transactional
    public Issued agent(String id, CanvasActor actor, AgentRunContext run) {
        var media = resources.get(id, false).resource();
        var limits = tickets.agentExpiry(run.runId(), actor.userId(), actor.teamId(), run.ownerInstanceId(),
                run.ownerEpoch(), media.projectId());
        if (limits == null) {
            throw denied();
        }
        return issue(id, actor, false,
                new Binding("AGENT", limits, null, null, run.runId(), run.ownerInstanceId(),
                        run.ownerEpoch(), null));
    }

    @Transactional
    public Issued canvasText(String id, CanvasActor actor, Claim claim) {
        var media = resources.get(id, false).resource();
        var limits = tickets.textExpiry(claim.id(), claim.owner(), claim.epoch(), actor.userId(), actor.teamId(),
                media.projectId());
        if (limits == null) {
            throw denied();
        }
        return issue(id, actor, false,
                new Binding("CANVAS_TEXT", limits, null, null, null, null, null, claim.id()));
    }

    private Issued issue(String id, CanvasActor actor, boolean download, Binding binding) {
        var resource = resources.get(id, false).resource();
        access.ready(resource, actor);
        byte[] random = new byte[32];
        RANDOM.nextBytes(random);
        String token = Base64.getUrlEncoder().withoutPadding().encodeToString(random);
        tickets.insert(MediaAccessTicketEntity.builder()
                .tokenHash(hash(token))
                .resourceId(id)
                .resourceVersion(resource.version())
                .actorUserId(actor.userId())
                .actorTeamId(actor.teamId())
                .purpose(binding.purpose())
                .download(download)
                .expiresAt(binding.expiresAt())
                .generationJobId(binding.jobId())
                .generationAttemptId(binding.attemptId())
                .agentRunId(binding.agentRunId())
                .agentOwnerInstanceId(binding.agentOwner())
                .agentOwnerEpoch(binding.agentEpoch())
                .canvasTextJobId(binding.textJobId()).build());
        return new Issued(id, "/api/media-content/" + id + "?ticket=" + token,
                binding.expiresAt().toInstant(ZoneOffset.UTC), download);
    }

    public Authorized authorize(String id, String token) {
        if (token == null || !token.matches("[A-Za-z0-9_-]{43}")) {
            throw denied();
        }
        var issued = tickets.authorize(hash(token), id);
        if (issued == null) {
            throw denied();
        }
        var ticket = new Ticket(new CanvasActor(issued.getActorUserId(), issued.getActorTeamId()),
                issued.getResourceVersion(), issued.getDownload(),
                new Binding(issued.getPurpose(), issued.getExpiresAt(),
                        issued.getGenerationJobId(), issued.getGenerationAttemptId(), issued.getAgentRunId(),
                        issued.getAgentOwnerInstanceId(), issued.getAgentOwnerEpoch(), issued.getCanvasTextJobId()));
        var row = resources.get(id, false);
        access.ready(row.resource(), ticket.actor());
        if (row.resource().version() != ticket.version()) {
            throw denied();
        }
        requireBinding(ticket, row.resource());
        return new Authorized(row.resource(), row.blobId(), ticket.actor(), ticket.download());
    }

    private void requireBinding(Ticket ticket, MediaResource resource) {
        var binding = ticket.binding();
        switch (binding.purpose()) {
            case "BROWSER" -> {
            }
            case "GENERATION" -> {
                long valid = tickets.validGeneration(binding.jobId(), binding.attemptId(), ticket.actor().userId(),
                        ticket.actor().teamId(), resource.projectId());
                if (valid != 1) {
                    throw denied();
                }
            }
            case "AGENT" -> {
                long valid = tickets.validAgent(binding.agentRunId(), ticket.actor().userId(), ticket.actor().teamId(),
                        binding.agentOwner(), binding.agentEpoch(), resource.projectId());
                if (valid != 1) {
                    throw denied();
                }
            }
            case "CANVAS_TEXT" -> {
                long valid = tickets.validText(binding.textJobId(), ticket.actor().userId(), ticket.actor().teamId(),
                        resource.projectId());
                if (valid != 1) {
                    throw denied();
                }
            }
            default -> throw denied();
        }
    }

    private String hash(String token) {
        try {
            return HexFormat.of()
                    .formatHex(MessageDigest.getInstance("SHA-256").digest(token.getBytes(StandardCharsets.US_ASCII)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    private MediaException denied() {
        return new MediaException(403, "MEDIA_TICKET_INVALID", "媒体访问凭据已失效，请重新获取");
    }

    private record Binding(String purpose, LocalDateTime expiresAt, String jobId, String attemptId, String agentRunId,
                           String agentOwner, Long agentEpoch, String textJobId) {
    }

    private record Ticket(CanvasActor actor, int version, boolean download, Binding binding) {
    }
}
