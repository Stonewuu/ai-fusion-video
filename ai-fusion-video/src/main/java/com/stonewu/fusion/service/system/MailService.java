package com.stonewu.fusion.service.system;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.mail.javamail.JavaMailSenderImpl;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import java.util.Properties;

/**
 * 邮件发送服务
 * 动态从数据库系统配置中读取配置，无需重启服务即可使配置生效
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class MailService {

    private final SystemConfigService systemConfigService;

    /**
     * 发送 HTML 邮件
     *
     * @param to      收件人邮箱
     * @param subject 邮件主题
     * @param content 邮件内容
     */
    public void sendHtmlEmail(String to, String subject, String content) {
        String host = systemConfigService.getValue("mail_smtp_host");
        String portStr = systemConfigService.getValue("mail_smtp_port");
        String username = systemConfigService.getValue("mail_username");
        String password = systemConfigService.getValue("mail_password");
        String sslStr = systemConfigService.getValue("mail_ssl");
        String from = systemConfigService.getValue("mail_from");

        if (StrUtil.isBlank(host) || StrUtil.isBlank(portStr) || StrUtil.isBlank(username) || StrUtil.isBlank(password)) {
            throw new BusinessException(400, "邮件服务未配置或配置不完整，请联系管理员配置邮箱参数");
        }

        int port;
        try {
            port = Integer.parseInt(portStr);
        } catch (NumberFormatException e) {
            throw new BusinessException(400, "邮件服务端口配置不正确，必须为数字");
        }

        boolean ssl = Boolean.parseBoolean(sslStr);

        JavaMailSenderImpl mailSender = new JavaMailSenderImpl();
        mailSender.setHost(host);
        mailSender.setPort(port);
        mailSender.setUsername(username);
        mailSender.setPassword(password);
        mailSender.setDefaultEncoding("UTF-8");

        Properties props = mailSender.getJavaMailProperties();
        props.put("mail.transport.protocol", "smtp");
        props.put("mail.smtp.auth", "true");
        // 对于主流协议（如 QQ, 163 等），开启 starttls
        props.put("mail.smtp.starttls.enable", "true");

        if (ssl) {
            props.put("mail.smtp.ssl.enable", "true");
            props.put("mail.smtp.socketFactory.class", "jakarta.net.ssl.SSLSocketFactory");
            props.put("mail.smtp.socketFactory.port", String.valueOf(port));
        }

        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
            helper.setTo(to);
            helper.setSubject(subject);
            helper.setText(content, true); // true 表示是 HTML 内容

            String fromAddress = StrUtil.isNotBlank(from) ? from : username;
            helper.setFrom(fromAddress);

            mailSender.send(message);
            log.info("邮件已成功发送至: {}", to);
        } catch (Exception e) {
            log.error("邮件发送失败, 收件人: {}", to, e);
            throw new BusinessException(500, "发送邮件失败: " + e.getMessage());
        }
    }
}
