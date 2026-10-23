package com.stonewu.fusion.service.media;

import com.google.crypto.tink.*;
import com.google.crypto.tink.streamingaead.PredefinedStreamingAeadParameters;
import com.google.crypto.tink.streamingaead.StreamingAeadConfig;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;

/**
 * 使用标准流式认证加密；存储端即使公开也只能得到密文。密钥只在服务端数据库保存。
 */
@Component
public class MediaCryptor {
    public MediaCryptor() {
        try {
            StreamingAeadConfig.register();
        } catch (GeneralSecurityException failure) {
            throw new IllegalStateException("媒体加密组件初始化失败", failure);
        }
    }

    public byte[] newKeyset() {
        try {
            var handle = KeysetHandle.generateNew(PredefinedStreamingAeadParameters.AES128_GCM_HKDF_1MB);
            return TinkProtoKeysetFormat.serializeKeyset(handle, InsecureSecretKeyAccess.get(),
                    RegistryConfiguration.get());
        } catch (GeneralSecurityException failure) {
            throw new IllegalStateException("媒体密钥生成失败", failure);
        }
    }

    public void encrypt(Path source, OutputStream destination, byte[] keyset, String resourceId) throws IOException {
        try (var output = primitive(keyset).newEncryptingStream(destination, context(resourceId))) {
            Files.copy(source, output);
        } catch (GeneralSecurityException failure) {
            throw new IOException("媒体加密失败", failure);
        }
    }

    public SeekableByteChannel decrypt(SeekableByteChannel source, byte[] keyset,
                                       String resourceId) throws IOException {
        try {
            var plaintext = primitive(keyset).newSeekableDecryptingChannel(source, context(resourceId));
            // Tink 先通过一次实际读取选择并认证密钥，之后才能取得明文长度。
            ByteBuffer first = ByteBuffer.allocate(1);
            int emptyReads = 0;
            while (plaintext.read(first) == 0) {
                if (++emptyReads > 8) {
                    throw new IOException("媒体读取未取得进展");
                }
            }
            plaintext.position(0);
            return plaintext;
        } catch (GeneralSecurityException | IOException | RuntimeException failure) {
            source.close();
            throw new IOException("媒体解密失败", failure);
        }
    }

    private StreamingAead primitive(byte[] keyset) throws GeneralSecurityException {
        return TinkProtoKeysetFormat.parseKeyset(keyset, InsecureSecretKeyAccess.get(), RegistryConfiguration.get())
                .getPrimitive(RegistryConfiguration.get(), StreamingAead.class);
    }

    private byte[] context(String id) {
        return ("afv-managed-media-v1:" + id).getBytes(StandardCharsets.UTF_8);
    }
}
