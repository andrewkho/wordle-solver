import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import 'bootstrap/dist/css/bootstrap.css';

import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import GoalWord from './GoalWord';
import Suggestor from './Suggestor';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<GoalWord/>} />
        <Route path="/suggestor" element={<Suggestor/>} />
      </Routes>
    </Router>
  );
}

export default App;
