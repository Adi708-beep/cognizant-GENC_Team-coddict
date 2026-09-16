import { Routes, Route } from 'react-router-dom'

import Sidebar from './components/Sidebar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import AddFeedback from './pages/AddFeedback'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />

      <Route
        path="/dashboard"
        element={
          <div className="min-h-screen bg-gray-50">
            <Sidebar />

            <main className="ml-64 min-h-screen p-8">
              <Dashboard />
            </main>
          </div>
        }
      />

      <Route
  path="/feedback"
  element={
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      <main className="ml-64 min-h-screen p-8">
        <AddFeedback />
      </main>
    </div>
  }
/>

      <Route
        path="/results"
        element={
          <div className="min-h-screen bg-gray-50">
            <Sidebar />

            <main className="ml-64 min-h-screen p-8">
              <h1 className="text-3xl font-bold text-gray-900">
                Results
              </h1>
            </main>
          </div>
        }
      />

      <Route
        path="/knowledge"
        element={
          <div className="min-h-screen bg-gray-50">
            <Sidebar />

            <main className="ml-64 min-h-screen p-8">
              <h1 className="text-3xl font-bold text-gray-900">
                Knowledge
              </h1>
            </main>
          </div>
        }
      />
    </Routes>
  )
}

export default App