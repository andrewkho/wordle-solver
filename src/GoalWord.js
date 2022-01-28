import React, { Component, useState, useEffect } from 'react';
import { Container, Form, FormControl, Button, Row, Col, ListGroup } from 'react-bootstrap';
import logo from './logo.svg';
import './App.css';

import NavBar from './NavBar';

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
            var resp = response.json();
            var msg = resp.msg;
            //var msg = response.text();
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
    <Container fluid="sm">
        <NavBar/>
    </Container>
        <br/>
        <div className='w-50 p-3' mw={578}>
        Enter a Goal Word
        <br/>
          <Form disabled={isLoading} onSubmit={!isLoading ? handleClick: null}>
            <Form.Group>
            <Row>
            <Col md={6} lg={8}>
              <Form.Control
                value={goalWord}
                onChange={e => setGoalWord(e.target.value)}
                placeholder={goalWord} />
            </Col>
            <Col md={6} lg={4}>
            <div className="d-grid gap-0">
            <Button
                variant="primary"
                disabled={isLoading}
                onClick={!isLoading ? handleClick: null}
            > {isLoading ? 'Waking up...' : 'Submit'} </Button>
            </div>
            </Col>
              </Row>
            </Form.Group>
          </Form>
          <br/>
        {renderResult()}
        <br/>
        <Button
          variant="secondary"
          href="/suggestor"> Want to me to suggest a word?</Button>
        </div>
      </center>
    );
  }

export default GoalWord;
