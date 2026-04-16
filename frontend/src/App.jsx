import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Visualise from './pages/Visualise.jsx'
import ClassificationPage from './pages/ClassificationPage.jsx'
import RegressionPage from './pages/RegressionPage.jsx'
import ClusteringPage from './pages/ClusteringPage.jsx'
import NeuralNetworkPage from './pages/NeuralNetworkPage.jsx'

export default function App() {
  return (
    <Routes>
      <Route path='/'              element={<Home />} />
      <Route path='/visualise'     element={<Visualise />} />
      <Route path='/classification' element={<ClassificationPage />} />
      <Route path='/regression'    element={<RegressionPage />} />
      <Route path='/clustering'    element={<ClusteringPage />} />
      <Route path='/neural-network' element={<NeuralNetworkPage />} />
    </Routes>
  )
}