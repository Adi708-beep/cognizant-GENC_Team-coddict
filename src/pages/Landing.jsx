import { ArrowRight, Sparkles, Upload, Brain, TrendingUp } from 'lucide-react'
import { Link } from 'react-router-dom'

function Landing() {
  return (
    <div className="min-h-screen bg-white">

      {/* Navigation */}
      <nav className="flex items-center justify-between px-8 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600">
            <Sparkles className="h-5 w-5 text-white" />
          </div>

          <span className="text-2xl font-bold text-gray-900">
            Feedy
          </span>
        </div>

        <Link
          to="/dashboard"
          className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700"
        >
          Open Dashboard
        </Link>
      </nav>

      {/* Hero */}
      <section className="mx-auto max-w-6xl px-8 pb-20 pt-16 text-center">

        <div className="mx-auto mb-6 flex w-fit items-center gap-2 rounded-full bg-indigo-50 px-4 py-2 text-sm font-medium text-indigo-600">
          <Sparkles className="h-4 w-4" />
          AI-Powered Feedback Intelligence
        </div>

        <h1 className="mx-auto max-w-4xl text-5xl font-bold tracking-tight text-gray-900 md:text-6xl">
          Turn Customer Feedback
          <span className="block text-indigo-600">
            Into Business Decisions
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-gray-600">
          Upload customer feedback, understand what your customers
          are saying, and turn their experiences into actionable insights.
        </p>

        <div className="mt-8 flex justify-center gap-4">
          <Link
            to="/feedback"
            className="flex items-center gap-2 rounded-xl bg-indigo-600 px-6 py-3 font-semibold text-white transition hover:bg-indigo-700"
          >
            Analyze Feedback
            <ArrowRight className="h-5 w-5" />
          </Link>

          <Link
            to="/dashboard"
            className="rounded-xl border border-gray-200 px-6 py-3 font-semibold text-gray-700 transition hover:bg-gray-50"
          >
            View Dashboard
          </Link>
        </div>
      </section>

      {/* Feature cards */}
      <section className="mx-auto grid max-w-6xl gap-6 px-8 pb-20 md:grid-cols-3">

        <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-50">
            <Upload className="h-5 w-5 text-indigo-600" />
          </div>

          <h3 className="text-lg font-semibold text-gray-900">
            Upload Anything
          </h3>

          <p className="mt-2 text-sm leading-6 text-gray-600">
            Upload CSV, Excel, PDF or image-based customer feedback.
          </p>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-50">
            <Brain className="h-5 w-5 text-indigo-600" />
          </div>

          <h3 className="text-lg font-semibold text-gray-900">
            AI Analysis
          </h3>

          <p className="mt-2 text-sm leading-6 text-gray-600">
            Detect sentiment, topics, severity and the reasons behind feedback.
          </p>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-50">
            <TrendingUp className="h-5 w-5 text-indigo-600" />
          </div>

          <h3 className="text-lg font-semibold text-gray-900">
            Actionable Insights
          </h3>

          <p className="mt-2 text-sm leading-6 text-gray-600">
            Discover recurring problems and get AI-powered recommendations.
          </p>
        </div>

      </section>

    </div>
  )
}

export default Landing