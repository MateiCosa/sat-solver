# Interpreter & Solver for Boolean Satisfiability Problems

<p align="middle">
    <img src="./img/title_img.jpg" alt = "Title Image">
</p>

## Scope 

[Boolean satifiability problems](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) (SAT) arise in many problems from the fields of theoretical computer science, optimization, game theory, and more. A famous result by Cook & Levin states that the Boolean satisfiability is [NP-complete](https://en.wikipedia.org/wiki/NP-completeness), meaning it takes exponential time to solve in the worst case. In practice, several tricks and heuristics may give rise to solvers that run quickly on relatively large instances.

Our goal is to build a SAT solver that takes an input in a specified format and identifies all the satisfying assignments (i.e., those for which the Boolean formula evaluates to True) of the input variables. To this end, we first fully specify the syntax of a declarative language in which the input must be written. Then we build a tokenizer, parser, and interpreter that executes the program and produces the desired output for correctly specified inputs. 

This project was completed as an assignment for the course [Software Engineering](https://www.poirrier.ca/courses/softeng/) taught by prof. Laurent Poirrier in the AI MSc @ Bocconi University. The problem description is directly based on the assignment question.

## Getting started

### Table of contents

* src: folder containing python files;
  * table.py: script to tokenize, parse, and compile programs;
* tests: folder containing two sub-folders;
  * demo: folder containing demo.txt example file;
  * test_instances: folders containing several (correct) test input files;
* requirements.txt: list of dependencies;
* img: folder containing title image;
* README.md: this file.

### Prerequisites

### Usage

### Example

The file `demo.txt` contains the following program:

`# We declare two variables:  x and y`
`var x y;`
`# We assign (x xor y) to z`
`z = (x or y) and (not (x and y));`
`# We show the truth table of z`
`show z;`

We run the following command:
`python3 src/table.py tests/demo/demo.txt`

and obtain the following output:
`0 0 0`
`0 1 1`
`1 0 0`
`1 1 0` 

where the columns correspond to the values of `x`, `y`, `z`, respectively.

## Input specification

We present the description of a declarative language that allows the creation of Boolean expressions, as well as their evaluation. Syntax rules and intruction descriptions are provided.

Note that any file not complying to this specification will result in an error message.

### BNF syntax

To formally describe the input specification, we introduce a BNF ([Backus–Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form)) syntax which fully explains the format of every valid input file (i.e., program) that our interpreter will parse and execute. 

| Object     | Description|
|------------|------------|
| \<element\>| "True" \| "False" \| \<identifier\>| 
| <paren-expr\>| \<element\> \| "(" \<expr\> ")"| 
| \<negation\>| "not" \<paren-expr\>|
| \<conjunction\>| \<paren-expr\> "and" \<paren-expr\> \| \<paren-expr\> "and" \<conjunction\>|
| \<disjunction\>| \<paren-expr\> "or" \<paren-expr\> \| \<paren-expr\> "or" \<disjunction\>|
| \<expr\>| \<negation\> \| \<conjunction\> \| \<disjunction\> \| \<paren-expr\>|
| \<id-list\>| \<identifier\> \| \<identifier\> \<id-list\>|
| \<instruction\>| "var" \<id-list\> ";" \| \<identifier\> "=" \<expr\> ";" \| "show" \<id-list\> ";" \| "show_ones" \<id-list\> ";"|
| \<program\>| \<instruction\> \| \<instruction\> \<program\>|

### Instructions

* **Variable declaration**: the keyword `var`, followed by (ii) a sequence of identifiers, then (iii) a semicolon (“;”). Identifiers cannot have been previously declared (neither as a variable, nor by an assignment).
* **Assignment**: (i) an identifier, followed by (ii) the special character “=” and (iii) an expression, then
(iv) a semicolon (“;”). The identifier in (i) cannot have been previously declared (neither as a variable, nor by an assignment). Before the assignment, every identifier present in the expression must have been either declared as a variable, or defined by an earlier assignment. In particular, the identifier in (i) cannot be used in the expression.
* **“Show” instruction**: (i) the keyword `show` or `show_ones`, followed by (ii) a list of identifiers, then (iii) a semicolon (“;”). Each identifier must have been defined by a prior assignment.

## Technical comments

This section provides a high-level summary of the overall implementation strategy, as well useful optimization techniques to improve performance.

### Expression parsing

Complex Boolean formulas can be viewed a nested sequence of expressions which is naturally representable through a stack. This is in fact our approach to model and parse the expressions in the input file. Here is an intuitive example:

`a = x and (y or (not z));`

Our stack contains a series of tokens for each level. At level 0, the stack stores `[x, and, x0]`, where `x0` is an abstraction for a Boolean expression. The next level contains `[y, or, y0]` where `y0` is an abstraction for a Boolean expression. Finally, the top of the stack is `[not, z]`. Notice how by evaluating the last expression we can pop the top of the stack and replace `y0` with the result of the previous level. Then we can repeat the same procedure for all remaining levels.

### Bit pattern generation via backtracking

Since we are interested in finding all possible variables assignments that result in a True evaluation, we need to generate all possible bit patterns of length equal to the number of variables. We do this through backtracking, as incremental bit generation can be very helpful for optimization tricks. Here is a simple example of bit generation for two variables `x` and `y`:

1. Set `x = 0` and move to the next variable.
   a. Set `y = 0`;
   b. Set `y = 1`.
2. Set `x = 1` and move to the next variable.
   a. Set `y = 0`;
   b. Set `y = 1`.

This approach is easily generalizable for a larger number of variables.

### Short-circuit evaluation

The key observation for optimizing our solution is that Boolean operators can be short-circuited. Consider the following example:

`a = x or True or y or z`

Suppose we are parsing this expression and find a True value in the disjunction. Then there is no point in evaluating the remaining variables and expressions. Similarly, the same applies for encountering a False value when evaluating a conjunction.

This idea plays perfectly into the backtracking scheme described before. After evaluating an expression by short-circuiting, the assignments of the remaining variables will give the same result, meaning there is no need to generate and evaluate them.

### Clause simplification

A final point which is essential for optimizing performance is keeping track of the sub-expressions already evaluated in a smart way. Here is a simple example:

`a = y or (not x)`

and suppose we set `x = 1`. Then we re-write the clause as `a = y or False`, which in turn can be reduced to `a = y`. 

These simplifcations make a huge difference for expressions that have many nested sub-expressions.