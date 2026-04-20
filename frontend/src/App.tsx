import { Routes, Route, Navigate } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import Nav from './components/ui/Nav'
import ErrorBoundary from './components/ui/ErrorBoundary'
import ScoutWidget from './components/scout/ScoutWidget'

const LotteryPage = lazy(() => import('./pages/LotteryPage'))
const DraftPage = lazy(() => import('./pages/DraftPage'))
const TeamPage = lazy(() => import('./pages/TeamPage'))
const ProspectsPage = lazy(() => import('./pages/ProspectsPage'))
const ScenarioPage = lazy(() => import('./pages/ScenarioPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="w-8 h-8 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-bg-primary">
        <Nav />
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<LotteryPage />} />
            <Route path="/lottery" element={<Navigate to="/" replace />} />
            <Route path="/draft" element={<DraftPage />} />
            <Route path="/teams/:teamId" element={<TeamPage />} />
            <Route path="/prospects" element={<ProspectsPage />} />
            <Route path="/scenario" element={<ScenarioPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
        <ScoutWidget />
      </div>
    </ErrorBoundary>
  )
}
