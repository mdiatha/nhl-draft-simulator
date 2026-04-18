import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  message: string
}

/**
 * Global error boundary — catches unhandled React render errors so the whole
 * app doesn't go blank. Shows a fallback UI with a reload button.
 *
 * Wrap around the Router in App.tsx.
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error.message ?? 'Unknown error' }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-bg-primary flex items-center justify-center px-4">
          <div className="max-w-md w-full bg-bg-card border border-border-default rounded-xl p-8 text-center">
            <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto mb-4">
              <span className="text-red-400 text-xl font-bold">!</span>
            </div>
            <h2 className="text-white text-xl font-bold mb-2">Something went wrong</h2>
            <p className="text-text-secondary text-sm mb-6">
              {this.state.message || 'An unexpected error occurred. Please reload the page.'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-accent-blue hover:bg-blue-500 text-white text-sm font-semibold rounded-lg transition-colors"
            >
              Reload page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
