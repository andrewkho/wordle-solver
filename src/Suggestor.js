import React, { Component, useState, useEffect } from 'react';
import { Container, Form, FormControl, Button, Row, Col, ListGroup } from 'react-bootstrap';
import logo from './logo.svg';
import './App.css';

import NavBar from './NavBar';

function processWordMasks(wordMasks) {
  const lines = wordMasks.split("\n");
  const words = [];
  const masks = [];
  for (let i = 0; i < lines.length && i<6; i++) {
    if (lines[i].length === 0) {
      break;
    }
    const wordMask = lines[i].split(" ");
    if (wordMask.length === 0) {
      break;
    }
    if (wordMask[0].length > 5 || wordMask[1].length > 5) {
      throw new Error("words or masks too long!")
    }
    words[i] = wordMask[0]
    masks[i] = wordMask[1]
  }

  return [words, masks]
}

function Suggestor() {
  const [isLoading, setLoading] = useState(false);
  const [wordMasks, setWordMasks] = useState("stare 20000\nclink 02222\n");
  const [suggestion, setSuggestion] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isLoading) {
      var url = "/api/wordle-suggest?";
      try {
        const [ words, masks ] = processWordMasks(wordMasks);
        url += "words=" + words.join(",");
        url += "&masks=" + masks.join(",");
        console.log(url);
      } catch {
        console.error("Problem processing wordMasks URL!");
        console.error(wordMasks);
        setError("Problem processing the words and masks, are they the " +
                 "right length and format?");
        return;
      }
      fetch(url)
      .then(response => {
        if (!response.ok) {
          setSuggestion("")
          setError("The server had a problem processing the words and masks, " +
                   "are they valid words and the right length and format?");
          console.error("Unknown error caught")
          throw new Error("Unknown error");
        }
        return response.json();
      }).then(resp => {
        setSuggestion(resp["suggestion"]);
        setError(null)
      })
      .catch(err => err)
      .finally(() => setLoading(false))
    }
  }, [isLoading]);

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
      Give me words and the masks from Wordle and see what I think is the next best move.
      Masks are 5 digits of 0, 1, 2's corresponding to No, Maybe, Yes responses from Wordle.
      <br/>
        <Form disabled={isLoading} onSubmit={!isLoading ? handleClick: null}>
          <Form.Group>
          <Row>
            <Form.Group className="mb-3" controlId="exampleForm.ControlTextarea1">
              <Form.Label>Enter Words and Masks</Form.Label>
                <p className="font-monospace">
                <Form.Control
                   as="textarea"
                   value={wordMasks}
                   onChange={e => setWordMasks(e.target.value)}
                   placeholder={wordMasks}
                   rows={6}/>
                </p>
              </Form.Group>
          <Col md={6} lg={8}>
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
        {error != null ? "Error! " + error : suggestion.length > 0 ? "I suggest " + suggestion: ""}
        <br/>
        <Button
          variant="secondary"
          href="/"> Want to try giving me a goal word? </Button>
      </div>
    </center>);
}

export default Suggestor;
