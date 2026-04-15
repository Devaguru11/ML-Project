import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Visualise from './pages/Visualise.jsx'
import ClassificationPage from './pages/ClassificationPage.jsx'
import RegressionPage from './pages/RegressionPage.jsx'
import ModelPage from './pages/ModelPage.jsx'
 
export default function App() {
 return (
   <Routes>
     <Route path='/' element={<Home />} />
     <Route path='/visualise' element={<Visualise />} />
     <Route path='/classification' element={<ClassificationPage />} />
     <Route path='/regression' element={<RegressionPage />} />
     <Route path='/clustering' element={<ModelPage type='clustering' />} />
     <Route path='/neural-network' element={<ModelPage type='neural-network' />} />
   </Routes>
 )
}
 