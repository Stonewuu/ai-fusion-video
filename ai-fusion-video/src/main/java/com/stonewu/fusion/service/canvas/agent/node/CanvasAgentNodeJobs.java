package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRuntimeInstanceIdentity;
import com.stonewu.fusion.service.ai.run.AgentStateSessionIds;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class CanvasAgentNodeJobs {
    private final CanvasAgentNodeRepository jobs;
    private final ObjectProvider<CanvasAgentNodePreparation> preparation;
    private final AgentConversationService conversations;
    private final AgentRunCoordinator coordinator;
    private final AgentRuntimeInstanceIdentity identity;
    private final AgentScopeV2Properties properties;
    private final CanvasJson json;

    @Transactional(propagation = Propagation.MANDATORY)
    @CacheEvict(value = "canvasAgentNode", key = "#node.id()")
    public String submit(Run run, NodeRun node) {
        var old = jobs.find(node.id(), false);
        if (old != null) {
            if (!old.canvasRunId().equals(run.id())) {
                throw CanvasRunException.conflict("专业智能体节点已经用于其他运行");
            }
            return old.agentRunId();
        }
        var prepared = preparation.getObject().prepare(run, node);
        String agentId = id(node.id(), "run");
        String conversationId = id(node.id(), "conversation");
        var type = prepared.scope().agentType();
        var deadline = AgentExecutionLimits.deadlineFrom(Instant.now(), properties.getExecution().getRunTimeout());
        if (deadline.isAfter(run.deadlineAt())) {
            deadline = run.deadlineAt();
        }
        conversations.createOrUpdate(conversationId, run.actor().userId(), run.plan().graph().projectId(), "project",
                run.plan().graph().projectId(), type, node.plan().node().title(), "canvas");
        jobs.insert(node.id(), run.id(), agentId, conversationId, prepared);
        coordinator.admitCanvasNode(
                new StartAgentRunCommand(agentId, conversationId, run.actor().userId(), run.plan().graph().projectId(),
                        type, null, null, null, AgentStateSessionIds.conversation(conversationId, type),
                        prepared.snapshot(), identity.value(),
                        properties.getExecution().getOwnerLease(), deadline, json.write(prepared.input()), null,
                        run.actor().teamId(), json.write(prepared.canvasScope())));
        return agentId;
    }

    private String id(String nodeId, String type) {
        return UUID.nameUUIDFromBytes(("canvas-agent:" + nodeId + ":" + type).getBytes(StandardCharsets.UTF_8))
                .toString().replace("-", "");
    }
}
