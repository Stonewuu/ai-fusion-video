package com.stonewu.fusion;

import com.stonewu.fusion.config.ApplicationTimeZoneInitializer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.context.WebServerInitializedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.EnableScheduling;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@SpringBootApplication
@EnableScheduling
public class FusionVideoApplication {

	public static void main(String[] args) {
		SpringApplication application = new SpringApplication(FusionVideoApplication.class);
		application.addInitializers(new ApplicationTimeZoneInitializer());
		application.run(args);
	}

	@EventListener
	public void onWebServerInitialized(WebServerInitializedEvent event) {
		int port = event.getWebServer().getPort();
		log.info("""
				\n==================================================
				Fusion Video 启动成功！
				服务地址: http://localhost:{}
				API 文档: http://localhost:{}/swagger-ui.html
				==================================================""", port, port);
	}

}
