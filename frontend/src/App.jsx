import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import ModelPage from './pages/ModelPage.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/classification" element={<ModelPage type="classification" />} />
      <Route path="/regression" element={<ModelPage type="regression" />} />
      <Route path="/clustering" element={<ModelPage type="clustering" />} />
      <Route path="/neural-network" element={<ModelPage type="neural-network" />} />
    </Routes>
  )
}
