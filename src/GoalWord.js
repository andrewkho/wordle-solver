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
  const [guesses, setGuesses] = useState(["test1", "test2", "test3"]);
  const [error, setError] = useState(null);
  const [won, setWon] = useState(false);

  useEffect(() => {
    if (isLoading) {
      fetch("/api/wordle-goal/" + goalWord)
      .then(response => {
        if (!response.ok) {
          setGuesses([])
          setWon(false)
          setError("Uh oh!")
          var msg = response.json().msg;
          setError(msg);
          throw new Error(msg);
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

  const handleClick = () => setLoading(true);

    return (
    <center>
    <div className="w-75 p-3">
        <NavBar/>
        <Container className="w-50 p-3">
        <br/>
        Enter a Goal Word
        <br/>
          <Row>
          <Col sm={9}>
          <Form>
            <Form.Group>
              <Form.Control
                  value={goalWord}
                  onChange={e => setGoalWord(e.target.value)}
              placeholder={goalWord} />
            </Form.Group>
          </Form>
            </Col>
            <Col sm={2}>
            <Button
              variant="primary"
              disabled={isLoading}
              onClick={!isLoading ? handleClick : null}
            >
              {isLoading ? 'Waking up...' : 'Submit'}
            </Button>
            </Col>
          </Row>
        {renderResult()}
        </Container>
      </div>
      </center>
    );
  }

export default GoalWord;
