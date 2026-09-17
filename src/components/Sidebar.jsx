import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  BarChart3,
  BookOpen,
  Sparkles,
} from 'lucide-react'

function Sidebar() {
  const navItems = [
    {
      name: 'Dashboard',
      path: '/dashboard',
      icon: LayoutDashboard,
    },
    {
      name: 'Add Feedback',
      path: '/feedback',
      icon: Upload,
    },
    {
      name: 'Results',
      path: '/results',
      icon: BarChart3,
    },
    {
      name: 'Knowledge',
      path: '/knowledge',
      icon: BookOpen,
    },
  ]

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 border-r border-gray-200 bg-white p-5">
      <div className="mb-10 flex items-center gap-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600">
          <Sparkles className="h-5 w-5 text-white" />
        </div>

        <span className="text-2xl font-bold text-gray-900">
          Feedy
        </span>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon

          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <Icon className="h-5 w-5" />
              {item.name}
            </NavLink>
          )
        })}
      </nav>
    </aside>
  )
}

export default Sidebar