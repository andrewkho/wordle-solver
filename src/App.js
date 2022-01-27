import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import 'bootstrap/dist/css/bootstrap.css';

import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import GoalWord from './GoalWord';
import Suggestor from './Suggestor';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      msg: "Not loaded"
    };
  }

  componentDidMount() {
    // Simple GET request using fetch
    fetch("/api/hello")
      .then(response => response.json())
      .then(data => this.setState({ msg: data.msg }));
  }

  render() {
    const { msg } = this.state;
    return (
      <Router>
        <Routes>
          <Route path="/" element={<GoalWord/>} />
          <Route path="/suggestor" element={<Suggestor/>} />
        </Routes>
      </Router>
    );
  }
}

//      <div className="App">
//        <div className="App-header">
//          <img src={logo} className="App-logo" alt="logo" />
//          <h2>Welcome to React</h2>
//        </div>
//        <p className="App-intro">
//          Hiiii To get started, edit <code>src/App.js</code> and save to reload. Got { msg } from flask
//        </p>
//        <GoalWord/>
//      </div>
export default App;
