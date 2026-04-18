import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <main className="max-w-lg mx-auto px-4 py-24 text-center">
      <p className="text-6xl font-black text-accent-blue mb-4">404</p>
      <h1 className="text-2xl font-bold text-white mb-2">Page not found</h1>
      <p className="text-text-secondary text-sm mb-8">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Link
        to="/"
        className="inline-block px-6 py-2 bg-accent-blue hover:bg-blue-500 text-white text-sm font-semibold rounded-lg transition-colors"
      >
        Back to home
      </Link>
    </main>
  )
}
