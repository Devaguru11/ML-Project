import { Routes, Route } from 'react-router-dom'

import LandingPage from './pages/LandingPage.jsx'
import UploadPage from './pages/UploadPage.jsx'
import DatasetReport from './pages/DatasetReport.jsx'
import Visualise from './pages/Visualise.jsx'
import ClassificationPage from './pages/ClassificationPage.jsx'
import RegressionPage from './pages/RegressionPage.jsx'
import ClusteringPage from './pages/ClusteringPage.jsx'
import NeuralNetworkPage from './pages/NeuralNetworkPage.jsx'

export default function App() {
  return (
    <Routes>
      <Route path='/' element={<LandingPage />} />
      <Route path='/upload' element={<UploadPage />} />
      <Route path='/dataset-report' element={<DatasetReport />} />
      <Route path='/visualise' element={<Visualise />} />
      <Route path='/classification' element={<ClassificationPage />} />
      <Route path='/regression' element={<RegressionPage />} />
      <Route path='/clustering' element={<ClusteringPage />} />
      <Route path='/neural-network' element={<NeuralNetworkPage />} />
    </Routes>
  )
}