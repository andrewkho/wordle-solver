import React, { Component } from 'react';
import { Container, NavDropdown, Navbar, Nav, Form, FormControl, Button, NavItem } from 'react-bootstrap';
import logo from './logo.svg';
import './App.css';

class NavBar extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
    <Container>
      <Navbar bg="light" expand="lg">
        <Container>
          <Navbar.Brand href="/">Wordle Deep-RL</Navbar.Brand>
          <Navbar.Toggle aria-controls="responsive-navbar-nav" />
          <Navbar.Collapse id="responsive-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link href="/">Goal Word</Nav.Link>
              <Nav.Link href="/suggestor">Suggestor</Nav.Link>
            </Nav>
          </Navbar.Collapse>
            <Nav className="me-auto">
              <Nav.Link href="https://andrewkho.github.io/wordle-solver">How this works</Nav.Link>
            </Nav>
        </Container>
      </Navbar>
    </Container>

    );
  }
}

export default NavBar;
