import type { ReactNode } from "react";

export default function AuthThemeLayout({ children }: { children: ReactNode }) {
  // 认证页面固定使用黑色背景；只限定此路由的控件主题，不改写用户的主题偏好。
  return <div className="dark scheme-dark text-foreground">{children}</div>;
}
