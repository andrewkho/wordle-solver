import React, { Component, useState, useEffect } from 'react';
import { Container, Form, FormControl, Button, Row, Col, ListGroup } from 'react-bootstrap';
import logo from './logo.svg';
import './App.css';

import NavBar from './NavBar';

function simulateNetworkRequest() {
  return new Promise((resolve) => setTimeout(resolve, 1000));
}

function formatResponse() {
}

function GoalWord() {
  const [isLoading, setLoading] = useState(false);
  const [goalWord, setGoalWord] = useState("slink");
  const [guesses, setGuesses] = useState([]);
  const [error, setError] = useState(null);
  const [won, setWon] = useState(false);

  useEffect(() => {
    if (isLoading) {
      fetch("/api/wordle-goal/" + goalWord)
      .then(response => {
        if (!response.ok) {
          setGuesses([])
          setWon(false)
          try {
            var msg = response.json().msg;
            setError(msg);
            throw new Error(msg);
          } catch {
            setError("Maybe it was invalid or not 5 chars?");
            console.error("Unknown error caught")
            throw new Error("Unknown error");
          }

        }
        return response.json();
      }).then(resp => {
        setWon(resp.win)
        setGuesses(resp.guesses)
        setError(null)
      })
      .catch(err => err)
      .finally(() => setLoading(false))
    }
  }, [isLoading]);

  const renderResult = () => {
    if (error != null) {
      return (
        <ListGroup>
          <ListGroup.Item>There was an error with your word! {error}</ListGroup.Item>
        </ListGroup>
      )
    }

    if (guesses.length > 0) {
      return (
        <div>
          <br/>
            I {(won? "won in " + guesses.length + " guesses!" : "lost!")}
          <ListGroup>
          {guesses.map(guess =>(
            <ListGroup.Item>{guess}</ListGroup.Item>
           ))}
          </ListGroup>
        </div>
      )
    } else {
      return (
        <ListGroup>
          <ListGroup.Item>Waiting...</ListGroup.Item>
        </ListGroup>
      )
    }
  };

  const handleClick = (e) => {
    e.preventDefault();
    setLoading(true);
  }

    return (
    <center>
    <div className="w-75 p-3">
        <NavBar/>
        <Container className="w-50 p-3">
        <br/>
        Enter a Goal Word
        <br/>
          <Form disabled={isLoading} onSubmit={!isLoading ? handleClick: null}>
            <Form.Group>
            <Row>
            <Col sm={8}>
              <Form.Control value={goalWord} onChange={e => setGoalWord(e.target.value)} placeholder={goalWord} />
            </Col>
            <Col sm={4}>
            <Button variant="primary" disabled={isLoading}> {isLoading ? 'Waking up...' : 'Submit'} </Button>
            </Col>
              </Row>
            </Form.Group>
          </Form>
        {renderResult()}
        </Container>
      </div>
      </center>
    );
  }

export default GoalWord;
