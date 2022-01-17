package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"runtime/debug"
	"strconv"
	"time"
)

func readWords(filename string) []Wordle {
	file, err := os.Open(filename)
	if err != nil {
		log.Panicln(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := []Wordle{}
	for i := 0; scanner.Scan(); i++ {
		line := scanner.Text()
		if len(line) != WordleN {
			log.Panicln(i, line)
		}
		lines = append(lines, Wordle(line))
	}

	if err := scanner.Err(); err != nil {
		log.Panicln(err)
	}
	return lines
}

const WordleN int = 5
type RuneOutcome int32
const (
	No RuneOutcome = iota
	Maybe RuneOutcome = iota
	Yes RuneOutcome = iota
)
type Result [WordleN]RuneOutcome
type Wordle string

func getResult(w0 Wordle, w1 Wordle) Result {
	result := Result{}
	Word0: 
	for i, c0 := range w0 {
		if rune(w1[i]) == c0 {
			result[i] = Yes
			continue
		}
		for j, c1 := range w1 {
			if c1 == c0 && j != i {
				result[i] = Maybe
				continue Word0
			}
		}
		result[i] = No
	}

	return result
}

func getOutcomes(word Wordle, testWords []Wordle) map[Result][]Wordle {
	results := make(map[Result][]Wordle)
	for _, test := range testWords {
		res := getResult(word, test)
		results[res] = append(results[res], test)
	}
	return results
}

func minimax(words []Wordle, testWords []Wordle) (Wordle, int, map[Result][]Wordle) {
	bst := 1<<63-1
	bstWord := Wordle("     ")
	bstOutcomes := map[Result][]Wordle{}
	for i, word := range words {
		if i % 1000 == 0 {
			fmt.Println("i:", i, word)
		}
		outcomes := getOutcomes(word, testWords)
		worst := 0
		for _, v := range outcomes {
			if len(v) > worst {
				worst = len(v)
			}
		}
		if worst <= bst {
			bst = worst
			bstWord = word
			bstOutcomes = outcomes
		}
	}

	return bstWord, bst, bstOutcomes
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(string(debug.Stack()))
		}
	}()

	var filename string = os.Args[1]
	t0 := time.Now()
	fmt.Printf("Reading from %s,", filename)
	fullWords := readWords(filename)
	fmt.Printf(" took %v for %v words\n", time.Since(t0), len(fullWords))

	result := Result{}
	outcomes := make(map[Result][]Wordle)
	outcomes[result] = fullWords
	for i := 0; true; i++ {
		t1 := time.Now()
		var bestWord Wordle
		score := 0
		testWords := outcomes[result]
		if len(testWords) == 1 {
			fmt.Printf("i: %v, Final word: %v\n", i, testWords[0])
			return
		} else if len(testWords) == 0 {
			fmt.Printf("i: %v, ran out of words!\n", i)
			return
		} else {
			candidates := fullWords
			if i == 0 {
				candidates = []Wordle{"serai"}
			}
			bestWord, score, outcomes = minimax(candidates, testWords)
			fmt.Printf("i: %v, %v, %v, %v, %v\n", i, result, bestWord, score, time.Since(t1))
		}

		fmt.Print("enter outcome (e.g. 12101): ")
		var input string
		fmt.Scanln(&input)
		fmt.Println("got", input)
		if len(input) < 5 {
			fmt.Println("Invalid input!")
			i--
		}
		result = Result{}
		for j, c := range input {
			if cint, err := strconv.Atoi(string(c)); err != nil {
				fmt.Println("Invalid input!", err)
			} else if cint < 0 || cint > 2 {
				fmt.Println("Invalid input!")
			} else {
				result[j] = RuneOutcome(cint)
			}
		}
	}
}