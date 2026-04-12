import { Routes, Route } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import Nav from './components/ui/Nav'

const HomePage = lazy(() => import('./pages/HomePage'))
const LotteryPage = lazy(() => import('./pages/LotteryPage'))
const DraftPage = lazy(() => import('./pages/DraftPage'))
const TeamPage = lazy(() => import('./pages/TeamPage'))
const ProspectsPage = lazy(() => import('./pages/ProspectsPage'))
const AdminPage = lazy(() => import('./pages/AdminPage'))
const ScoutPage = lazy(() => import('./pages/ScoutPage'))
const ModelPage = lazy(() => import('./pages/ModelPage'))

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="w-8 h-8 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-bg-primary">
      <Nav />
      <Suspense fallback={<PageLoader />}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lottery" element={<LotteryPage />} />
          <Route path="/draft" element={<DraftPage />} />
          <Route path="/teams/:teamId" element={<TeamPage />} />
          <Route path="/prospects" element={<ProspectsPage />} />
          <Route path="/scout" element={<ScoutPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </Suspense>
    </div>
  )
}
