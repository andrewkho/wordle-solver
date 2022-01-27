import React, { Component, useState, useEffect } from 'react';
import { Container, Form, FormControl, Button, Row, Col} from 'react-bootstrap';
import logo from './logo.svg';
import './App.css';

import NavBar from './NavBar';

function simulateNetworkRequest() {
  return new Promise((resolve) => setTimeout(resolve, 1000));
}

  function GoalWord() {
    const [isLoading, setLoading] = useState(false);
    const [goalWord, setGoalWord] = useState("slink");
    const [guesses, setGuesses] = useState("guesses");

    useEffect(() => {
      if (isLoading) {
        fetch("/api/wordle-goal/" + goalWord)
        .then(response => {
          console.log(response.status);
          if (!response.ok) {
            setGuesses("Error!" + response.msg);
            throw new Error(response.msg);
          }
          return response.json()
         }).then(resp => {
              var output_text = '';
              if (resp.win) {
                output_text += 'I won in ' + resp.guesses.length + ' guesses!\n';
              } else {
                output_text += 'I lost!\n';
              }
              for (var i in resp.guesses) {
                output_text += resp.guesses[i] + '\n';
              }
              setGuesses(output_text)
              console.log(output_text)
        })
        .catch(err => err)
        .finally(() => setLoading(false))
      }
      console.log("Hi!");
    }, [isLoading]);

    const handleClick = () => setLoading(true);

    return (
    <div>
        <NavBar/>
        <Container>
        Enter a Goal Word
        <Row className="mb-3">
          <Col>
          <Form>
            <Form.Group
                className="mb-3"
                controlId="exampleForm.ControlInput1">
              <Form.Control
                  value={goalWord}
                  onChange={e => setGoalWord(e.target.value)}
              placeholder={goalWord} />
            </Form.Group>
          </Form>
          </Col>
          <Col>
            <Button
              variant="primary"
              disabled={isLoading}
              onClick={!isLoading ? handleClick : null}
            >
              {isLoading ? 'Waking up...' : 'Submit'}
            </Button>
          </Col>
        </Row>
        {guesses}
        </Container>
      </div>
    );
  }

export default GoalWord;
