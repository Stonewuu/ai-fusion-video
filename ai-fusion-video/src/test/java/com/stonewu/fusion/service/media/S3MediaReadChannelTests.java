package com.stonewu.fusion.service.media;

import org.junit.jupiter.api.Test;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.http.AbortableInputStream;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.HeadObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectResponse;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class S3MediaReadChannelTests {
    @Test
    void abortsWhenStorageIgnoresRangeBeforeReadingTheBody() throws Exception {
        var client = mock(S3Client.class);
        when(client.headObject(anyConsumer())).thenReturn(HeadObjectResponse.builder().contentLength(20L).eTag("fixed").build());
        AtomicBoolean aborted = new AtomicBoolean();
        AtomicBoolean read = new AtomicBoolean();
        var input = new ByteArrayInputStream(new byte[20]) {
            @Override public synchronized int read(byte[] bytes, int offset, int length) {
                read.set(true);
                return super.read(bytes, offset, length);
            }
        };
        when(client.getObject(any(GetObjectRequest.class))).thenReturn(new ResponseInputStream<>(
                GetObjectResponse.builder().contentLength(20L).build(), AbortableInputStream.create(input, () -> aborted.set(true))));
        try (var channel = new S3MediaReadChannel(client, "bucket", "object", 20)) {
            assertThatThrownBy(() -> channel.read(ByteBuffer.allocate(10))).isInstanceOf(IOException.class);
        }
        assertThat(aborted).isTrue();
        assertThat(read).isFalse();
    }
    @SuppressWarnings("unchecked")
    private Consumer<HeadObjectRequest.Builder> anyConsumer() { return any(Consumer.class); }
}
