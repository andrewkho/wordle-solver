import React, { Component } from 'react';
import { Container, Form, FormControl, Button } from 'react-bootstrap';
import logo from './logo.svg';
import './App.css';

import NavBar from './NavBar';

class GoalWord extends React.Component {
  constructor(props) {
    super(props);
    this.state = {value: ''};

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({value: event.target.value});
  }

  handleSubmit(event) {
    alert('A name was submitted: ' + this.state.value);
    event.preventDefault();
  }

  render() {
    return (
      <NavBar/>
    );
  }
}

export default GoalWord;
