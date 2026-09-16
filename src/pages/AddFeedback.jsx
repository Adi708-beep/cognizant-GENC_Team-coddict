import { useState } from 'react'
import { uploadFeedback } from '../api/api'
import { Upload, FileText, CheckCircle } from 'lucide-react'

function AddFeedback() {
  const [file, setFile] = useState(null)

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0]

    if (selectedFile) {
      setFile(selectedFile)
    }
  }
  const handleUpload = async () => {
  if (!file) return

  try {
    await uploadFeedback(file)
    alert('Feedback uploaded successfully!')
  } catch (error) {
    console.error(error)
    alert('Upload failed. Please check that the backend is running.')
  }
}

  return (
    <div className="mx-auto max-w-4xl">

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Add Feedback
        </h1>

        <p className="mt-2 text-gray-600">
          Upload customer feedback for AI-powered analysis.
        </p>
      </div>

      {/* Upload Card */}
      <div className="rounded-2xl bg-white p-8 shadow-sm">

        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900">
            Upload Feedback File
          </h2>

          <p className="mt-1 text-sm text-gray-500">
            Select a CSV file containing your customer feedback.
          </p>
        </div>

        {/* Upload Area */}
        <label
          htmlFor="feedback-file"
          className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 px-6 py-12 transition hover:border-indigo-400 hover:bg-indigo-50"
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-indigo-100">
            <Upload className="h-7 w-7 text-indigo-600" />
          </div>

          <p className="mt-4 text-lg font-medium text-gray-900">
            Click to upload
          </p>

          <p className="mt-1 text-sm text-gray-500">
            CSV files only
          </p>

          <input
            id="feedback-file"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>

        {/* Selected File */}
        {file && (
          <div className="mt-6 flex items-center justify-between rounded-xl border border-green-200 bg-green-50 p-4">

            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white">
                <FileText className="h-5 w-5 text-green-600" />
              </div>

              <div>
                <p className="font-medium text-gray-900">
                  {file.name}
                </p>

                <p className="text-sm text-gray-500">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>

            <CheckCircle className="h-6 w-6 text-green-600" />
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!file}
          className="mt-6 w-full rounded-xl bg-indigo-600 px-5 py-3 font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          Upload Feedback
        </button>

      </div>

      {/* Info */}
      <div className="mt-6 rounded-2xl border border-indigo-100 bg-indigo-50 p-6">
        <h3 className="font-semibold text-indigo-900">
          How it works
        </h3>

        <div className="mt-4 space-y-3 text-sm text-indigo-800">
          <p>1. Upload your customer feedback CSV.</p>
          <p>2. Feedy analyzes the feedback.</p>
          <p>3. View sentiment, issues, and recommendations on the dashboard.</p>
        </div>
      </div>

    </div>
  )
}

export default AddFeedback