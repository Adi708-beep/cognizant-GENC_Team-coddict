import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

function Dashboard() {
  const sentimentData = [
    { name: 'Positive', value: 150 },
    { name: 'Neutral', value: 45 },
    { name: 'Negative', value: 55 },
  ]

  const issues = [
    { name: 'Waiting Time', count: 42 },
    { name: 'Service Quality', count: 31 },
    { name: 'Food Quality', count: 24 },
    { name: 'Pricing', count: 18 },
  ]

  const recommendations = [
    'Reduce customer waiting time during peak hours.',
    'Improve staff response and service quality.',
    'Review recurring complaints about food quality.',
  ]

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Dashboard
        </h1>

        <p className="mt-2 text-gray-600">
          Understand what your customers are saying.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-2xl bg-white p-6 shadow-sm">
          <p className="text-sm text-gray-500">Total Feedback</p>
          <h2 className="mt-2 text-3xl font-bold text-gray-900">
            250
          </h2>
        </div>

        <div className="rounded-2xl bg-white p-6 shadow-sm">
          <p className="text-sm text-gray-500">Positive</p>
          <h2 className="mt-2 text-3xl font-bold text-green-600">
            150
          </h2>
        </div>

        <div className="rounded-2xl bg-white p-6 shadow-sm">
          <p className="text-sm text-gray-500">Neutral</p>
          <h2 className="mt-2 text-3xl font-bold text-yellow-500">
            45
          </h2>
        </div>

        <div className="rounded-2xl bg-white p-6 shadow-sm">
          <p className="text-sm text-gray-500">Negative</p>
          <h2 className="mt-2 text-3xl font-bold text-red-500">
            55
          </h2>
        </div>
      </div>

      {/* Chart + Issues */}
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">

        {/* Sentiment Chart */}
        <div className="rounded-2xl bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900">
            Sentiment Distribution
          </h2>

          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sentimentData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={100}
                  paddingAngle={4}
                >
                  {sentimentData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        index === 0
                          ? '#22c55e'
                          : index === 1
                          ? '#eab308'
                          : '#ef4444'
                      }
                    />
                  ))}
                </Pie>

                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Issues */}
        <div className="rounded-2xl bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900">
            Top Issues
          </h2>

          <div className="mt-5 space-y-5">
            {issues.map((issue, index) => (
              <div key={issue.name}>
                <div className="mb-2 flex justify-between">
                  <span className="font-medium text-gray-700">
                    {index + 1}. {issue.name}
                  </span>

                  <span className="text-sm text-gray-500">
                    {issue.count}
                  </span>
                </div>

                <div className="h-2 rounded-full bg-gray-100">
                  <div
                    className="h-2 rounded-full bg-indigo-500"
                    style={{
                      width: `${(issue.count / 42) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* AI Recommendations */}
      <div className="mt-6 rounded-2xl bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900">
          AI Recommendations
        </h2>

        <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-3">
          {recommendations.map((recommendation, index) => (
            <div
              key={index}
              className="rounded-xl border border-gray-100 bg-gray-50 p-5"
            >
              <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-100 text-sm font-bold text-indigo-600">
                {index + 1}
              </div>

              <p className="text-sm leading-6 text-gray-600">
                {recommendation}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Dashboard