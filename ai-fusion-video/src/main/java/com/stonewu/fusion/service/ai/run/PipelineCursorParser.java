package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.run.model.RunCursor;
import org.springframework.stereotype.Component;

/** Strictly reconciles query and standard Last-Event-ID cursors. */
@Component
public final class PipelineCursorParser {

    public RunCursor parse(
            String authorizedRunId,
            Long queryAfterSequence,
            String lastEventId) {
        String runId = requireRunId(authorizedRunId);
        if (queryAfterSequence != null && queryAfterSequence < 0) {
            throw invalid("afterSequence must not be negative");
        }
        if (lastEventId == null) {
            return cursor(runId, queryAfterSequence == null ? 0 : queryAfterSequence);
        }
        if (lastEventId.isBlank() || !lastEventId.equals(lastEventId.trim())) {
            throw invalid("Last-Event-ID is invalid");
        }

        int separator = lastEventId.lastIndexOf(':');
        if (separator <= 0 || separator == lastEventId.length() - 1) {
            throw invalid("Last-Event-ID is invalid");
        }
        String headerRunId = lastEventId.substring(0, separator);
        if (!runId.equals(headerRunId)) {
            throw invalid("Last-Event-ID targets a different run");
        }
        long headerSequence = parseSequence(lastEventId.substring(separator + 1));
        if (queryAfterSequence != null
                && queryAfterSequence.longValue() != headerSequence) {
            throw invalid("afterSequence conflicts with Last-Event-ID");
        }
        return cursor(runId, headerSequence);
    }

    private String requireRunId(String runId) {
        try {
            return new RunCursor(runId, 0).runId();
        } catch (IllegalArgumentException invalid) {
            throw invalid(invalid.getMessage());
        }
    }

    private long parseSequence(String encoded) {
        if (encoded.isEmpty()) {
            throw invalid("Last-Event-ID sequence is missing");
        }
        for (int index = 0; index < encoded.length(); index++) {
            char digit = encoded.charAt(index);
            if (digit < '0' || digit > '9') {
                throw invalid("Last-Event-ID sequence is invalid");
            }
        }
        try {
            return Long.parseLong(encoded);
        } catch (NumberFormatException overflow) {
            throw invalid("Last-Event-ID sequence is too large");
        }
    }

    private RunCursor cursor(String runId, long sequence) {
        try {
            return new RunCursor(runId, sequence);
        } catch (IllegalArgumentException invalid) {
            throw invalid(invalid.getMessage());
        }
    }

    private BusinessException invalid(String message) {
        return new BusinessException(400, message);
    }
}
