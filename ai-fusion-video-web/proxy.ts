import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// 不需要认证的路径
const PUBLIC_PATHS = ["/login", "/register", "/setup", "/forgot-password"];

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 静态资源和 API 路由不拦截
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api") ||
    pathname.startsWith("/favicon") ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // 从 cookie 中读取认证状态（zustand persist 存在 localStorage，middleware 无法直接读取）
  // 使用一个轻量的 cookie 来同步认证状态
  const token = request.cookies.get("auth-token")?.value;
  const isPublicPath = PUBLIC_PATHS.some((path) => pathname.startsWith(path));
  const isRoot = pathname === "/";

  // 已认证访问根路径 → 直接跳 dashboard（跳过初始化检测）
  if (token && isRoot) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  // 根路径放行（未认证时需要客户端检测初始化状态）
  if (isRoot) {
    return NextResponse.next();
  }

  // 未认证访问受保护页面 → 重定向到登录
  if (!token && !isPublicPath) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  // 已认证访问登录页 → 重定向到面板
  if (token && isPublicPath) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    // 匹配所有路径，排除静态资源
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
