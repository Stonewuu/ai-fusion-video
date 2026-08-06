"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";
import { notifyClientError } from "@/lib/client-error";

interface ClientErrorBoundaryProps {
  children: ReactNode;
  context: string;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ClientErrorBoundaryState {
  error: Error | null;
}

export class ClientErrorBoundary extends Component<
  ClientErrorBoundaryProps,
  ClientErrorBoundaryState
> {
  state: ClientErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ClientErrorBoundaryState {
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
