package com.stonewu.fusion.service.ai.run;

/** Projects AgentScope 2.0.0 tool-result blocks using their actual error contract. */
final class AgentScopeToolResultStatus {

    private static final String ERROR_PREFIX = "Error: ";

    private AgentScopeToolResultStatus() {
    }

    static String project(String state, String result) {
        if (isErrorState(state) || isOfficialErrorBlock(result)) {
            return "error";
        }
        return "success";
    }

    private static boolean isErrorState(String state) {
        return state != null
                && ("ERROR".equalsIgnoreCase(state) || "FAILED".equalsIgnoreCase(state));
    }

    private static boolean isOfficialErrorBlock(String result) {
        return result != null && result.startsWith(ERROR_PREFIX);
    }
}
