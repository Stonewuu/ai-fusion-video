"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";
import { notifyClientError } from "@/lib/client-error";

interface ClientErrorBoundaryProps {
  children: ReactNode;
  context: string;
  /** 切换页面时清除旧错误，同时保留正常子树的组件状态。 */
  resetKey?: string;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ClientErrorBoundaryState {
  error: Error | null;
  resetKey?: string;
}

export class ClientErrorBoundary extends Component<
  ClientErrorBoundaryProps,
  ClientErrorBoundaryState
> {
  state: ClientErrorBoundaryState = {
    error: null,
    resetKey: this.props.resetKey,
  };

  static getDerivedStateFromProps(
    props: ClientErrorBoundaryProps,
    state: ClientErrorBoundaryState
  ): Partial<ClientErrorBoundaryState> | null {
    if (props.resetKey !== state.resetKey) {
      return { error: null, resetKey: props.resetKey };
    }
    return null;
  }

  static getDerivedStateFromError(error: Error): Partial<ClientErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[${this.props.context}]`, error, info.componentStack);
    notifyClientError(error, this.props.context);
    this.props.onError?.(error, info);
  }

  private reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }
    return typeof this.props.fallback === "function"
      ? this.props.fallback(error, this.reset)
      : this.props.fallback ?? null;
  }
}
