package com.stonewu.fusion.security;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.CorsProperties;
import com.stonewu.fusion.controller.media.MediaContentController;
import com.stonewu.fusion.controller.media.MediaExceptionHandler;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.media.MediaAccessTickets;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.model.MediaException;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.mock.web.MockServletContext;
import org.springframework.security.web.FilterChainProxy;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.context.support.AnnotationConfigWebApplicationContext;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.head;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class MediaSecurityBoundaryTests {
    @Configuration
    @EnableWebMvc
    static class Web {
        @Bean ObjectMapper mapper() { return new ObjectMapper(); }
        @Bean CorsProperties cors() { return new CorsProperties(); }
        @Bean TokenAuthenticationFilter tokens() {
            return new TokenAuthenticationFilter(mock(TokenService.class), mock(SecurityUserDetailsService.class));
        }
        @Bean MediaAccessTickets tickets() {
            var tickets = mock(MediaAccessTickets.class);
            when(tickets.authorize(anyString(), anyString())).thenThrow(new MediaException(403, "MEDIA_TICKET_INVALID", "无效凭据"));
            return tickets;
        }
        @Bean MediaContentController content(MediaAccessTickets tickets) {
            return new MediaContentController(tickets, mock(MediaResourceRepository.class), mock(MediaBlobStore.class));
        }
        @Bean MediaExceptionHandler advice() { return new MediaExceptionHandler(); }
    }

    @Test
    void onlyGetAndHeadReachTheTicketGateWhileCreationAndAccessIssuanceRequireLogin() throws Exception {
        try (var context = new AnnotationConfigWebApplicationContext()) {
            context.setServletContext(new MockServletContext());
            context.register(Web.class, SecurityConfig.class);
            context.refresh();
            var http = MockMvcBuilders.webAppContextSetup(context)
                    .addFilters(context.getBean(FilterChainProxy.class)).build();
            http.perform(get("/api/media-content/test-id").param("ticket", "bad")).andExpect(status().isForbidden());
            verify(context.getBean(MediaAccessTickets.class)).authorize("test-id", "bad");
            http.perform(head("/api/media-content/test-id").param("ticket", "bad")).andExpect(status().isForbidden());
            http.perform(post("/api/media-content/test-id").param("ticket", "bad")).andExpect(status().isUnauthorized());
            http.perform(post("/api/media-resources/test-id/access")).andExpect(status().isUnauthorized());
            http.perform(post("/api/media-resources/upload")).andExpect(status().isUnauthorized());
            http.perform(get("/api/media-resources/test-id")).andExpect(status().isUnauthorized());
        }
    }
}
