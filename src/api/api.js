import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

export const uploadFeedback = (file) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/feedback/upload', formData)
}

export const getDashboard = () => {
  return api.get('/feedback/dashboard')
}

export const getResults = () => {
  return api.get('/feedback/results')
}

export const uploadKnowledge = (file) => {
  const formData = new FormData()
  formData.append('file', file)

  return api.post('/knowledge/upload', formData)
}

export const getKnowledge = () => {
  return api.get('/knowledge/list')
}

export const sendChatMessage = (message) => {
  return api.post('/chat', { message })
}

export default api