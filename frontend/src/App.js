import React from 'react'
import '../node_modules/bootstrap/dist/css/bootstrap.min.css'
import './App.css'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import SignUp from './components/signup'
import Login from './components/Login'
import HomePage from './components/HomePage';
function App() {
  return (
    <Router>
            <Routes>
              <Route path="/" element={<Login />} />
              <Route exact path="/dashboard" element={<HomePage />} />
              <Route path="/sign-up" element={<SignUp />} />
            </Routes>
    </Router>
  )
}
export default App