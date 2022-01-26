import React, { Component } from 'react';
import GoalWord from './GoalWord.js';
import logo from './logo.svg';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      msg: "Not loaded"
    };
  }

  componentDidMount() {
    // Simple GET request using fetch
    fetch("/hello")
      .then(response => response.json())
      .then(data => this.setState({ msg: data.msg }));
  }

  render() {
    const { msg } = this.state;
    return (
      <div className="App">
        <div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h2>Welcome to React</h2>
        </div>
        <p className="App-intro">
          Hiiii To get started, edit <code>src/App.js</code> and save to reload. Got { msg } from flask
        </p>
        <GoalWord/>
      </div>
    );
  }
}

export default App;
