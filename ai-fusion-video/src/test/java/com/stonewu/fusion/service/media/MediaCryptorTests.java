package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Random;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MediaCryptorTests {
    @TempDir Path directory;

    @Test
    void authenticatesSegmentsAndAllowsSeekingAcrossBoundaries() throws Exception {
        var cryptor = new MediaCryptor();
        byte[] plaintext = new byte[3 * 1024 * 1024 + 73];
        new Random(61).nextBytes(plaintext);
        Path source = directory.resolve("input");
        Path output = directory.resolve("encrypted");
        Files.write(source, plaintext);
        byte[] key = cryptor.newKeyset();
        cryptor.encrypt(source, Files.newOutputStream(output), key, "resource-1");
        assertThat(Files.size(output)).isGreaterThan(plaintext.length);
        assertThat(Arrays.equals(Files.readAllBytes(output), plaintext)).isFalse();
        try (var decoded = cryptor.decrypt(Files.newByteChannel(output), key, "resource-1")) {
            assertThat(decoded.size()).isEqualTo(plaintext.length);
            for (int start : new int[]{0, 1024 * 1024 - 17, 2 * 1024 * 1024 + 4, plaintext.length - 33, 71}) {
                int length = Math.min(3111, plaintext.length - start);
                ByteBuffer buffer = ByteBuffer.allocate(length);
                decoded.position(start);
                while (buffer.hasRemaining()) assertThat(decoded.read(buffer)).isPositive();
                assertThat(buffer.array()).isEqualTo(Arrays.copyOfRange(plaintext, start, start + length));
            }
        }
        assertThatThrownBy(() -> {
            try (var decoded = cryptor.decrypt(Files.newByteChannel(output), key, "other-resource")) {
                decoded.read(ByteBuffer.allocate(32));
            }
        }).isInstanceOf(IOException.class);
        try (var corrupt = Files.newByteChannel(output, StandardOpenOption.WRITE)) {
            corrupt.position(1200000);
            corrupt.write(ByteBuffer.wrap(new byte[]{123, 19, 7, 1, 85}));
        }
        assertThatThrownBy(() -> {
            try (var decoded = cryptor.decrypt(Files.newByteChannel(output), key, "resource-1")) {
                decoded.position(1100000);
                decoded.read(ByteBuffer.allocate(32));
            }
        }).isInstanceOf(IOException.class);
    }

    @Test
    void rangeHandlesVideoSeekSuffixAndInvalidInputs() {
        assertThat(MediaByteRange.parse(null, 20)).isEqualTo(new MediaByteRange(0, 19, false));
        assertThat(MediaByteRange.parse("bytes=7-", 20)).isEqualTo(new MediaByteRange(7, 19, true));
        assertThat(MediaByteRange.parse("bytes=-6", 20)).isEqualTo(new MediaByteRange(14, 19, true));
        assertThat(MediaByteRange.parse("bytes=5-99", 20)).isEqualTo(new MediaByteRange(5, 19, true));
        for (String invalid : new String[]{"bytes=-0", "bytes=-", "bytes=20-", "bytes=8-2", "bytes=1-2,7-8",
                "items=1-3", "bytes=999999999999999999999999-", "bytes=-1-2"}) {
            assertThatThrownBy(() -> MediaByteRange.parse(invalid, 20)).isInstanceOf(MediaException.class);
        }
    }
}
