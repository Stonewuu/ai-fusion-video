package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.configuration.AgentDefinitionConfiguration;
import com.stonewu.fusion.service.ai.configuration.AgentSettingsOverride;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasAgentProfiles {
    public record Profile(String type, String name, List<CanvasValueType> inputTypes, boolean singleInput,
                          boolean requiresTypedInput, List<CanvasValueType> resultTypes, String hint,
                          boolean customized, boolean toolsEnabled, List<String> defaultSkills,
                          List<String> generationKinds) {
    }

    private final AiAgentService agents;

    public List<Profile> list(long userId) {
        return CanvasAgentNodeContracts.list().stream().map(contract -> {
            var variables = Map.of(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE,
                    agents.snapshot(contract.type(), userId));
            var definition = agents.fromSnapshot(contract.type(), variables);
            var tools = CanvasAgentNodeTypes.tools(type -> agents.fromSnapshot(type, variables), definition);
            var kinds = new ArrayList<String>();
            if (tools.contains("generate_image")) {
                kinds.add("image");
            }
            if (tools.contains("generate_video")) {
                kinds.add("video");
            }
            return new Profile(contract.type(), definition.getName(), contract.inputTypes(), contract.singleInput(),
                    contract.requiresTypedInput(), contract.resultTypes(), contract.hint(),
                    !agents.snapshotOverride(contract.type(), variables).equals(AgentSettingsOverride.inherit()),
                    Integer.valueOf(1).equals(definition.getEnableTools()), definition.getDefaultSkills(),
                    List.copyOf(kinds));
        }).toList();
    }
}
