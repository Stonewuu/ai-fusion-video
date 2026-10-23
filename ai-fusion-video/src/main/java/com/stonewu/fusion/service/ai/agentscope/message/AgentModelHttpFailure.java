package com.stonewu.fusion.service.ai.agentscope.message;

import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;

import io.agentscope.core.model.ModelHttpException;
import io.agentscope.core.model.transport.HttpTransportException;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * An HTTP error response is a finished request, unlike a transport disconnect.
 */
public final class AgentModelHttpFailure {
    private AgentModelHttpFailure() {
    }

    public static Integer status(Throwable failure) {
        var rejected = GenerationProviderFailure.find(failure);
        if (rejected != null) {
            return rejected.httpStatus();
        }
        Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Throwable current = failure; current != null && seen.add(current); current = current.getCause()) {
            Integer status = current instanceof ModelHttpException response ? response.getStatusCode()
                    : current instanceof HttpTransportException transport ? transport.getStatusCode() : null;
            if (status != null && status >= 400 && status < 600) {
                return status;
            }
        }
        return null;
    }
}
