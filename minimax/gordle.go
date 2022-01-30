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

func getResult(guess Wordle, evalWord Wordle) Result {
	result := Result{}
	Word0: 
	for i, c0 := range guess {
		if rune(evalWord[i]) == c0 {
			result[i] = Yes
			continue
		}
		for j, c1 := range evalWord {
			if c1 == c0 && j != i {
				result[i] = Maybe
				continue Word0
			}
		}
		result[i] = No
	}

	return result
}

func getOutcomes(guess Wordle, remainingWords []Wordle) map[Result][]Wordle {
	results := make(map[Result][]Wordle)
	for _, evalWord := range remainingWords {
		res := getResult(guess, evalWord)
		results[res] = append(results[res], evalWord)
	}
	return results
}

func minimax(candidates []Wordle, remainingWords []Wordle, verbose bool) (Wordle, int, map[Result][]Wordle) {
	bst := 1<<63-1
	bstGuess := Wordle("     ")
	bstOutcomes := map[Result][]Wordle{}
	for i, guess := range candidates {
		if verbose && (i % 1000 == 0) {
			fmt.Println("i:", i, guess)
		}
		outcomes := getOutcomes(guess, remainingWords)
		worst := 0
		for _, v := range outcomes {
			if len(v) > worst {
				worst = len(v)
			}
		}
		if worst <= bst {
			bst = worst
			bstGuess = guess
			bstOutcomes = outcomes
		}
	}

	return bstGuess, bst, bstOutcomes
}

func playGame(goal_word Wordle, fullWords []Wordle, evalWords []Wordle) (bool, int) {
	result := Result{}
	outcomes := make(map[Result][]Wordle)
	outcomes[result] = evalWords

	for turns := 0; turns < 6; turns++ {
		var bstGuess Wordle
		remainingWords := outcomes[result]
		if len(remainingWords) == 1 {
			return true, turns+1
		} else if len(remainingWords) == 0 {
			return false, -1
		} 
		candidates := fullWords
		if turns == 0 {
			candidates = []Wordle{"serai"}
		}
		bstGuess, _, outcomes = minimax(candidates, remainingWords, false)
		
		result = getResult(bstGuess, goal_word)
	}

	return false, 6
}

func evaluate(fullWords []Wordle, evalWords []Wordle) {
	fmt.Printf("Evaluation mode for %v evaluation words\n", len(evalWords))
	wins := 0
	N := len(evalWords)
	n_guesses := 0
	n_win_guesses := 0
	for i, goal_word := range evalWords {
		if i % 100 == 0 {
			fmt.Println("step", i)
		}
		win, guesses := playGame(goal_word, fullWords, evalWords)
		if win {
			wins++
			n_win_guesses += guesses
		} else {
			fmt.Println("Lost!", goal_word)
		}
		n_guesses += guesses
	}

	fmt.Printf("Evaluation complete, won %v %% of games, average %v guesses for wins, %v guesses in total\n",
		  float64(wins) / float64(N), float64(n_win_guesses) / float64(wins), float64(n_guesses) / float64(N))
}

func play(fullWords []Wordle, evalWords []Wordle) {
	result := Result{}
	outcomes := make(map[Result][]Wordle)
	outcomes[result] = evalWords
	for i := 0; true; i++ {
		t1 := time.Now()
		var bestWord Wordle
		score := 0
		remainingWords := outcomes[result]
		if len(remainingWords) == 1 {
			fmt.Printf("i: %v, Final word: %v\n", i, remainingWords[0])
			return
		} else if len(remainingWords) == 0 {
			fmt.Printf("i: %v, ran out of words!\n", i)
			return
		} else {
			candidates := fullWords
			if i == 0 {
				candidates = []Wordle{"serai"}
			}
			bestWord, score, outcomes = minimax(candidates, remainingWords, true)
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

func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(string(debug.Stack()))
		}
	}()

	filename := os.Args[1]
	mode := os.Args[2]
	t0 := time.Now()
	fmt.Printf("Reading from %s,", filename)
	fullWords := readWords(filename)
	evalWords := fullWords[:2315]
	fmt.Printf(" took %v for %v words\n", time.Since(t0), len(fullWords))
	fmt.Printf("Mode: %v\n", mode)

	if mode == "play" {
		play(fullWords, evalWords)
	} else {
		evaluate(fullWords, evalWords)

	}
}