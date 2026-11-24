package com.stonewu.fusion.service.ai.run.stream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Range;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.data.redis.connection.Limit;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.time.Instant;
import java.util.LinkedHashSet;

/** 运行中的流完整保留；结束后流和索引使用相同的固定过期时间。 */
@Component
@RequiredArgsConstructor
public class AgentRedisStreamStore {
    private static final DefaultRedisScript<Long> APPEND = new DefaultRedisScript<>("""
            if ARGV[3] == 'existing' and redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
            local previous = redis.call('HGET', KEYS[3], ARGV[1])
            if previous then return tonumber(previous) end
            local sequence = redis.call('INCR', KEYS[2])
            redis.call('XADD', KEYS[1], tostring(sequence) .. '-0', 'data', ARGV[2])
            redis.call('HSET', KEYS[3], ARGV[1], tostring(sequence))
            return sequence
            """, Long.class);
    private static final DefaultRedisScript<Long> ADVANCE_CONTROL = new DefaultRedisScript<>("""
            if redis.call('EXISTS', KEYS[2]) == 0 then return 0 end
            local current = tonumber(redis.call('HGET', KEYS[1], ARGV[1]) or '0')
            local sequence = tonumber(ARGV[2])
            if sequence > current then redis.call('HSET', KEYS[1], ARGV[1], ARGV[2]) end
            local remaining = redis.call('PTTL', KEYS[2])
            if remaining >= 0 then redis.call('PEXPIRE', KEYS[1], remaining) end
            return math.max(current, sequence)
            """, Long.class);
    private static final DefaultRedisScript<Long> EXPIRE = new DefaultRedisScript<>("""
            for _, key in ipairs(KEYS) do
                redis.call('PEXPIREAT', key, ARGV[1])
            end
            return 1
            """, Long.class);

    private final StringRedisTemplate redis;
    private final ObjectMapper json;

    public AiChatStreamRespVO append(String runId, String identity, AiChatStreamRespVO event) {
        return append(runId, identity, event, false);
    }

    public AiChatStreamRespVO appendExisting(String runId, String identity, AiChatStreamRespVO event) {
        return append(runId, identity, event, true);
    }

    private AiChatStreamRespVO append(String runId, String identity, AiChatStreamRespVO event, boolean existing) {
        String key = key(runId);
        event.setSchemaVersion(2).setRunId(runId).setSequence(null);
        Long sequence = redis.execute(APPEND, List.of(key, key + ":sequence", key + ":identities"),
                identity, encode(event), existing ? "existing" : "create");
        if (existing && Long.valueOf(0).equals(sequence)) {
            return event;
        }
        if (sequence == null || sequence <= 0) {
            throw new IllegalStateException("消息流写入未返回有效序号");
        }
        return event.setSequence(sequence);
    }

    public List<AiChatStreamRespVO> readAfter(String runId, long afterSequence, int limit) {
        var range = Range.of(Range.Bound.exclusive(afterSequence + "-0"), Range.Bound.<String>unbounded());
        List<MapRecord<String, String, String>> records = redis.<String, String>opsForStream()
                .range(key(runId), range, Limit.limit().count(limit));
        List<AiChatStreamRespVO> result = new ArrayList<>();
        if (records == null) {
            return result;
        }
        for (MapRecord<String, String, String> record : records) {
            AiChatStreamRespVO event = decode(record.getValue().get("data"));
            event.setSequence(Long.parseLong(record.getId().getValue().split("-", 2)[0]));
            result.add(event);
        }
        return result;
    }

    public long lastSequence(String runId) {
        String sequence = redis.opsForValue().get(key(runId) + ":sequence");
        return sequence == null ? 0 : Long.parseLong(sequence);
    }

    public long controlSequence(String streamRunId, String sourceRunId) {
        Object sequence = redis.opsForHash().get(key(streamRunId) + ":controls", sourceRunId);
        return sequence == null ? 0 : Long.parseLong(sequence.toString());
    }

    public void advanceControl(String streamRunId, String sourceRunId, long sequence) {
        redis.execute(ADVANCE_CONTROL, List.of(key(streamRunId) + ":controls", key(streamRunId)), sourceRunId,
                Long.toString(sequence));
    }

    public boolean exists(String runId) {
        return Boolean.TRUE.equals(redis.hasKey(key(runId)));
    }

    public void expireAt(String runId, Instant deadline) {
        String key = key(runId);
        redis.execute(EXPIRE, List.of(key, key + ":sequence", key + ":identities", key + ":controls"),
                Long.toString(deadline.toEpochMilli()));
    }

    public List<String> existingRuns() {
        var runs = new LinkedHashSet<String>();
        String prefix = "fv:ai:run:{";
        String suffix = "}:events";
        try (var cursor = redis.scan(ScanOptions.scanOptions().match(prefix + "*" + suffix).count(256).build())) {
            while (cursor.hasNext()) {
                String key = cursor.next();
                runs.add(key.substring(prefix.length(), key.length() - suffix.length()));
            }
        }
        return List.copyOf(runs);
    }

    public Long eventSequence(String streamRunId, String identity) {
        Object sequence = redis.opsForHash().get(key(streamRunId) + ":identities", identity);
        return sequence == null ? null : Long.parseLong(sequence.toString());
    }

    private String key(String runId) {
        return "fv:ai:run:{" + runId + "}:events";
    }

    private String encode(AiChatStreamRespVO event) {
        try {
            return json.writeValueAsString(event);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("消息流序列化失败", error);
        }
    }

    private AiChatStreamRespVO decode(String value) {
        if (value == null) {
            throw new IllegalStateException("消息流记录缺少内容");
        }
        try {
            return json.readValue(value, AiChatStreamRespVO.class);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("消息流内容无效", error);
        }
    }
}
