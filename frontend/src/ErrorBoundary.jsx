import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('Unhandled UI error:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: '32px',
            fontFamily: 'monospace',
            color: '#e9e4d8',
            background: '#12181f',
            height: '100vh',
          }}
        >
          <h2 style={{ fontFamily: 'serif', fontWeight: 400 }}>
            Something went wrong
          </h2>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#a9b3bd' }}>
            {this.state.error?.message}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: '12px',
              background: '#d98f3a',
              color: '#241505',
              border: 'none',
              padding: '10px 18px',
              borderRadius: '2px',
              cursor: 'pointer',
            }}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
