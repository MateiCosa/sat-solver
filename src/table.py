import sys
import re
import numpy as np
import os

# ----------------------- global variables -----------------------
special_chars = set(['(', ')', '=', ';']) # special characters
instructions = ['var', 'show', 'show_ones'] # instructions
bool_ops = ['not', 'and', 'or'] # boolean operators
bool_vals = [False, True] # boolean values
program = [] # tokenized sequence of keywords, identifiers, and special characters
var_list = [] # list of variable names in order of appearance
var_table = None # table with all truth values of the variables
expr_list = [] # list of identifier names corresponding to expressions (assignments) in order of appearance
expr_eval_set = set() # identifiers that need to evaluated
expr_table = None # table to store values of the expressions
expr_mapping = {} # name(str): expr (object)
instr_list = [] # list of instr
true_patterns = [] # list of patterns that yield True
# ----------------------------------------------------------------
# ---------------------------- utils -----------------------------
def rec_expr(expr):
    '''
    Parses expr and finds present ids recursively.

    Args:
        expr (Expr): expression to parse.

    Returns: 
        None.
    '''
    global expr_eval_set

    if not expr.expr_set:
        return
    
    for sub_expr in expr.expr_set:
        expr_eval_set.add(sub_expr)
        rec_expr(expr_mapping[sub_expr])
    
def gen_table(n):
    '''
    Generates truth table for n variables in lexicographic order.

    Args:
        n (int): total number of variables in the program.

    Returns: 
        np.array((2**n, n), dtype=bool).
    '''
    return np.array([[c == '1' for c in s] for s in np.vectorize(np.binary_repr)(np.arange(2**n), n)])

def gen_combinations(n):
    '''
    Generates vector containing bit patterns for n variables in lexicographic order.

    Args:
        n (int): number of variables.

    Returns: 
        np.array(2**n, dtype=str)
    '''
    return np.vectorize(np.binary_repr)(np.arange(2**n), n)

def gen_row_patterns(table, cols):
    '''
    Generates vector containing bit patterns for variables in cols from table.

    Args:
        table (np.array): table of truth values;
        cols (list(int)): list of column indices to extract from table.

    Returns: 
        np.array(2**n, dtype=str)
    '''
    return np.array([''.join([str(c) for c in row ]) for row in (table[:, cols]*1)], dtype=np.str_)

def eval_rec_level(clause, truth_val_mapping):
    '''
    Evaluates the clause formed by the given by the recursion level using 
    the truth values mapping.

    Args:
        clause (str): boolean clause (top of stack) to be evaluated;
        truth_value_mapping (dict): mapping from variables to truth values.

    Returns:
        bool: result of the boolean expression.
    '''
    result = None
    clause_type = -1 # {-1: unknown, 0: val/id, 1: neg, 2: conj, 3: disj}

    if len(clause) == 1: # truth val or id
        result = eval_expr(clause[0], truth_val_mapping)
        clause_type = 0 # expression

    elif len(clause) == 2: # negation
        result = not eval_expr(clause[1], truth_val_mapping)
        clause_type = 1 # negation

    else: # conjunction or disjunction
        clause_type = 2 if clause[1] == 'and' else 3
        for i, component in enumerate(clause):
            
            if component == 'or' or component == 'and':
                continue

            val = eval_expr(component, truth_val_mapping)

            if clause_type == 2 and val == False:
                result = False
                break
            if clause_type == 3 and val == True:
                result = True
                break

            if result is None: # initialization
                result = val
            else:
                result = (result and val) if clause_type == 2 else (result or val) # update result
    
    return result

def eval_expr(keyword, truth_val_mapping):
    '''
    Evaluates the keyword (boolean value, variable, expr id).

    Args:
        keyword (str or bool): boolean value, variable, expr id;
        truth_value_mapping (dict): mapping from variables to truth values.

    Returns:
        bool: truth value for the keyword.
    '''
    if keyword == True or keyword == 'True':
        val = True
    elif keyword == False or keyword == 'False':
        val = False
    elif keyword in truth_val_mapping: # variable
        val = truth_val_mapping[keyword]
    else: # missed some identifier
        raise Exception("Did not substitute all identifiers!")
    return val

def optim_eval_rec_level(clause):
    '''
    Evaluates the clause formed by the given recursion level.
    Short-circuiting + clause simplification.

    Args:
        clause (str): boolean clause (top of stack) to be evaluated;

    Returns:
        bool/list: result of the boolean expression.
    '''
    result = None
    clause_type = -1 # {-1: unknown, 0: val/id, 1: neg, 2: conj, 3: disj}

    if len(clause) == 1: # truth val or id
        clause_type = 0 # expression
        clause = clause[0]

    elif len(clause) == 2: # negation
        clause_type = 1 # negation
        if clause[1] in bool_vals:
            clause = not clause[1]

    else: # conjunction or disjunction
        clause_type = 2 if clause[1] == 'and' else 3
        new_clause = []
        for i in range(0, len(clause), 2):

            if clause_type == 2:
                if clause[i] == False:
                    new_clause = False
                    break
                elif clause[i] == True:
                    continue
                else: # id
                    if len(new_clause) >= 1:
                        new_clause.append('and')
                        new_clause.append(clause[i])
                    else:
                        new_clause.append(clause[i])

            if clause_type == 3:
                if clause[i] == True:
                    new_clause = True
                    break
                elif clause[i] == False:
                    continue
                else: # id
                    if len(new_clause) >= 1:
                        new_clause.append('or')
                        new_clause.append(clause[i])
                    else:
                        new_clause.append(clause[i])
        if new_clause not in bool_vals and len(new_clause) == 0:
            if clause_type == 2:
                new_clause = True
            else:
                new_clause = False
        clause = new_clause

    while type(clause) == list and len(clause) == 1:
        clause = clause[0]
    
    return clause

def backtrack(expr_body, var_list, actual_var_set, pattern):
    '''
    Builds true patterns incrementally.

    Args:
        expr_body (str): boolean clause to be evaluated;
        var_list (list): list of vars that need to show up in truth table;
        actual_var_set (set): set of vars present in expr_body;
        pattern (str): current bit pattern.

    Returns:
        None.
    '''
    global true_patterns 

    if not var_list: # used all variables
        new_expr = Expr('0', 0, set(), set(), expr_body)
        if new_expr.eval({}):
            true_patterns.append(pattern)
        return

    if var_list[0] not in actual_var_set: # variable does not show up in the expression
        backtrack(expr_body, var_list[1:], actual_var_set, pattern + '0')
        backtrack(expr_body, var_list[1:], actual_var_set, pattern + '1')
        return

    for val in bool_vals:
        body_copy = expr_body.copy()
        enc = '1' if val else '0'
        result = expr_subst(body_copy, var_list[0], val)
        if result in bool_vals:
            if result == True:
                if len(var_list) == 1:
                    true_patterns.append(pattern + enc)
                else:
                    for new_pattern in gen_table(len(var_list)-1):
                        true_patterns.append(pattern + enc +''.join([str(i) for i in new_pattern*1]))
        else:
            backtrack(result, var_list[1:], actual_var_set, pattern + enc)

def expr_subst(body, var, val):
    '''
    Subsitutes literal var with its value val.

    Args:
        body (str): boolean clause;
        var (str): literal id;
        val (bool): value of literal.

    Returns:
        bool/list: result of the boolean expression.
    '''

    length = len(body)
    cursor = 0
    stack = [[]]

    while cursor < length:
        if body[cursor] == var:
            body[cursor] = val
        token = body[cursor]
        
        if token == '(': # start new recursion level
            stack.append([])

        elif token == ')': # eval recursion level

            result = optim_eval_rec_level(stack[-1])
            #if type(result) == list and len(result) > 1:
            #    result = 'paren-expr'
            
            stack.pop()
            if stack:
                    stack[-1].append(result)
            else:
                stack.append([result])

        else: # append token to the current recursion level
            stack[-1].append(token)
            
        cursor += 1

    if stack:
        result = optim_eval_rec_level(stack[-1])
    
    if result in bool_vals:
        return result
    else:
        return body

def merge(a, b):
    '''
    Merges two sorted (in increasing order) lists.

    Args:
        a (list): list representing pattern;
        b (list): list representing pattern;

    Returns:
        list: merged list.
    '''
    n, m = len(a), len(b)
    i = j = 0
    c = []

    while i < n and j < m:
        if a[i] == b[j]:
            c.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    
    while i < n:
        c.append(a[i])
        i += 1
    while j < m:
        c.append(b[j])
        j += 1

    return c
# ----------------------------------------------------------------

class Tokenizer():

    def __init__(self, file_name):
        self.file_name = file_name # input file name
        
    def tokenize_line(self, line):
        invalid_char = False
        ignore_ending = False

        no_white_space_list = [token for token in line.strip().split(' ') if token] # removes white spaces 
        tokens_list = [] # store the tokens
        for token in no_white_space_list:

            if ignore_ending: break # check if we encountered invalid character

            if re.match(r'^\w+$', token): # check if token contains alphanum and _ only
                if (ord(token[0]) - ord('0')) >= 0 and (ord(token[0]) - ord('0')) <= 9:
                    raise Exception("Identifiers cannot start with a digit!")
                tokens_list.append(token)
            else:
                start = 0
                for i, c in enumerate(token):
                    if c.isalnum() or c == '_':
                        continue
                    if c == '#': # comment
                        if i > start:
                            if (ord(token[start]) - ord('0')) >= 0 and (ord(token[start]) - ord('0')) <= 9:
                                raise Exception("Identifiers cannot start with a digit!")
                            tokens_list.append(token[start:i])
                        ignore_ending = True
                        break
                    if c in special_chars: # special char
                        if start < i: 
                            if (ord(token[start]) - ord('0')) >= 0 and (ord(token[start]) - ord('0')) <= 9:
                                raise Exception("Identifiers cannot start with a digit!")
                            tokens_list.append(token[start:i])
                        tokens_list.append(c)
                        start = i + 1
                        continue
                    invalid_char = True # invalid char
                    break

                if invalid_char:
                    raise Exception(f"Invalid character '{c}' in the input!")
                
                if not ignore_ending and start < len(token): # add the remaining chars (if any) and no comment
                    tokens_list.append(token[start:])

        for i, token in enumerate(tokens_list): # change type of bool values from str to bool
            if token == 'True':
                tokens_list[i] = True
            elif token == 'False':
                tokens_list[i] = False

        return tokens_list
    
    def run(self):
        global program
        file_name = self.file_name
        with open(file_name) as file:
            for line in file:
                tokens_list = self.tokenize_line(line)
                if tokens_list:
                    program.extend(tokens_list)

class Parser():

    def __init__(self):
        self.length = len(program)
        self.cursor = 0
    
    def run(self):
        global expr_list, expr_mapping

        while self.cursor < self.length:
            token = program[self.cursor]
            if token == 'var':
                self.parse_declaration()
            elif token == 'show' or token == 'show_ones':
                instr_type = 0 if token == 'show' else 1
                self.parse_print(instr_type)
            else:
                if token in var_list or token in expr_list:
                    raise Exception(f"Identifier '{token}' assigned after prior declaration or assignment!")
                if token in bool_ops or token in bool_vals or token in special_chars:
                    raise Exception(f"Keyword '{token}' is not a valid instruction!")
                self.parse_assignment()
        
        for instr in instr_list: # check which identifiers show up in show instructions
            for expr_name in instr[1]: # list of args (ids)
                expr_eval_set.add(expr_name)
                expr = expr_mapping[expr_name]
                rec_expr(expr)

        true_expr_list = []
        for expr_name in expr_list:
            if expr_name in expr_eval_set:
                true_expr_list.append(expr_name)
                body = expr_mapping[expr_name].body
                new_body = []
                for token in body:
                    if token in expr_list:
                        new_body.append('(')
                        new_body.extend(expr_mapping[token].body)
                        new_body.append(')')
                    else:
                        new_body.append(token)
                expr_mapping[expr_name].body = new_body
        expr_list = true_expr_list
            
    def parse_declaration(self):

        global var_list
        self.cursor += 1
        found_id = False

        while self.cursor < self.length and program[self.cursor] != ';':
            token = program[self.cursor]

            if token in instructions or token in bool_ops or token in bool_vals or token in special_chars:
                raise Exception(f"Keyword '{token}' is not a valid identifier!")
            if token in var_list or token in expr_list:
                raise Exception(f"Identifier '{token}' assigned after prior declaration or assignment!")
            
            found_id = True
            var_list.append(token)

            if len(var_list) > 64:
                raise Exception("Found more than 64 identifiers!")
            
            self.cursor += 1

        if self.cursor >= self.length:
            raise Exception("Expected ';' at the end of variable declaration!")
        
        if not found_id:
            raise Exception("Expected list of variables after 'var'!")
        
        self.cursor += 1

    def parse_assignment(self):

        global expr_list, expr_mapping
    
        # identifier for the expression
        id = program[self.cursor]
        if id in instructions or id in bool_ops or id in bool_vals:
            raise Exception(f"Keyword '{id}' is not a valid identifier!")
        if id in var_list or id in expr_list:
            raise Exception(f"Identifier '{id}' assigned after prior declaration or assignment!")
        
        # equal sign
        self.cursor += 1
        if program[self.cursor] != '=':
            raise Exception(f"Expected '=' after identifier in expression!")
        
        # variables that make up the expression
        self.cursor += 1
        expr_start = self.cursor
        var_set = set()
        expr_set = set()
        stack = [[]]
        while self.cursor < self.length and program[self.cursor] != ';':
            token = program[self.cursor]

            if token in instructions: # no instructions within expression
                raise Exception("Expected expression for assignment!")
            
            if token == '(':
                stack[-1].append("paren-expr")
                stack.append([])

            elif token == ')':

                if len(stack) <= 1:
                    raise Exception("Parenthesis closed without being open!")

                if not len(stack[-1]):
                    raise Exception("Empty paren-expr!")
                
                conjunction = False
                disjunction = False
                negation = False
                for i, component in enumerate(stack[-1]):
                    if component == "and": 
                        if i == 0 or stack[-1][i-1] != "paren-expr" or i == len(stack[-1])-1 or stack[-1][i+1] != "paren-expr":
                            raise Exception("Invalid syntax for conjunction!")
                        conjunction = True
                    elif component == "or":
                        if i == 0 or stack[-1][i-1] != "paren-expr" or i == len(stack[-1])-1 or stack[-1][i+1] != "paren-expr":
                            raise Exception("Invalid syntax for disjunction!")
                        disjunction = True
                    elif component == "not":
                        if len(stack[-1]) > 2 or len(stack[-1]) == 1:
                            raise Exception("Invalid expression. Expected '(not ...)'!")
                        negation = True
                
                if (not (conjunction or disjunction or negation)) and len(stack[-1]) > 1:
                    raise Exception("Invalid expression. Expected boolean operator!")

                if conjunction and disjunction:
                    raise Exception("Expected '(...)' to separate conjunctions and disjunctions!")
                
                stack.pop()

            elif token == "not":
                if len(stack[-1]):
                    raise Exception("Invalid expression. Expected '(not paren-expr)'!")
                else:
                    stack[-1].append("not")

            elif token == "and":
                stack[-1].append("and")

            elif token == "or":
                stack[-1].append("or")

            elif token in bool_vals:
                stack[-1].append("paren-expr")

            else: # must be an identifier!
                if token not in var_list and token not in expr_list:
                    raise Exception(f"Identifier {token} not declared or assigned prior to expression!")
                
                stack[-1].append("paren-expr")
                if token in var_list: # var
                    var_set.add(token)
                else: # id
                    var_set = var_set.union(expr_mapping[token].var_set)
                    expr_set.add(token)
            
            self.cursor += 1
        
        if self.cursor >= self.length:
            raise Exception("Expected ';' at the end of identifier assignment!")

        if len(stack) > 1:
            raise Exception("Parenthesis left unclosed!")
        
        if stack:
            if not len(stack[-1]):
                    raise Exception("Empty paren-expr!")

            conjunction = False
            disjunction = False
            negation = False
            for i, component in enumerate(stack[-1]):
                if component == "and": 
                    if i == 0 or stack[-1][i-1] != "paren-expr" or i == len(stack[-1])-1 or stack[-1][i+1] != "paren-expr":
                        raise Exception("Invalid syntax for conjunction!")
                    conjunction = True
                elif component == "or":
                    if i == 0 or stack[-1][i-1] != "paren-expr" or i == len(stack[-1])-1 or stack[-1][i+1] != "paren-expr":
                        raise Exception("Invalid syntax for disjunction!")
                    disjunction = True
                elif component == "not":
                    if len(stack[-1]) > 2 or len(stack[-1]) == 1:
                        raise Exception("Invalid expression. Expected '(not paren-expr)'!")
                    negation = True
                
            if (not (conjunction or disjunction or negation)) and len(stack[-1]) > 1:
                raise Exception("Invalid expression. Expected boolean operator!")

            if conjunction and disjunction:
                raise Exception("Expected '(...)' to separate conjunctions and disjunctions!")
       
        expr_list.append(id)
        expr_mapping[id] = Expr(id, len(expr_list) - 1, var_set, expr_set, program[expr_start:self.cursor])
        self.cursor += 1

    def parse_print(self, instr_type):

        global instr_list

        self.cursor += 1
        arg_list = []

        found_id = False
        
        while self.cursor < self.length and program[self.cursor] != ';':
            token = program[self.cursor]

            if token in instructions or token in bool_ops or token in bool_vals or token in special_chars:
                raise Exception(f"Keyword '{token}' is not a valid identifier!")
            if token in var_list:
                raise Exception(f"Keyword '{token}' assigned to variable, not identifier!")
            if token not in expr_list:
                raise Exception(f"Identifier '{token}' used without prior declaration or assignment!")
            
            arg_list.append(token)
            found_id = True
            
            self.cursor += 1

        if self.cursor >= self.length:
            raise Exception("Expected ';' at the end of variable declaration!")

        if not found_id:
            raise Exception("Expected list of identifiers after printing instruction!")
        
        self.cursor += 1

        instr_list.append((instr_type, arg_list, len(var_list))) # (instr_type, variables, last var index plus one)
    
class Compiler():

    def __init__(self):
        pass

    def run(self):
        for instr in instr_list:
            instr_type, arg_list, var_index = instr
            if instr_type == 0:
                self.execute_show(arg_list, var_index)
            else:
                self.execute_show_ones(arg_list, var_index)
    
    def execute_show(self, arg_list, var_index):

        global true_patterns
        
        var_table = gen_table(var_index) * 1 # all input combinations
        table = np.hstack((var_table, np.zeros((len(var_table), len(arg_list)), dtype=int)))

        for i, arg in enumerate(arg_list):
            
            true_patterns = []
            backtrack(expr_mapping[arg].body, var_list[:var_index], expr_mapping[arg].var_set, '')
            ones = [int(pattern, 2) for pattern in true_patterns]
            table[:, var_index + i][ones] = 1

        np.savetxt(sys.stdout.buffer, table, fmt="%.0d") 

    def execute_show_ones(self, arg_list, var_index):

        global true_patterns

        show_patterns = None
        cache = [set() for arg in arg_list]

        for i, arg in enumerate(arg_list):
            
            true_patterns = []
            backtrack(expr_mapping[arg].body, var_list[:var_index], expr_mapping[arg].var_set, '')

            # caching vals
            for pattern in true_patterns:
                cache[i].add(pattern)

            # merge current pattern list with the previous one (both are sorted!)
            if show_patterns is None:
                show_patterns = true_patterns
            else:
                show_patterns = merge(show_patterns, true_patterns)

        # create output table
        table = np.zeros((len(show_patterns), var_index + len(arg_list)), dtype = bool)
        truth_val_mapping = dict()
        for i, pattern in enumerate(show_patterns):
            table[i, :var_index] = [c == '1' for c in pattern]
            for j in range(var_index):
                truth_val_mapping[var_list[j]] = (pattern[j] == '1')
            for j, arg in enumerate(arg_list):
                if pattern in cache[j]:
                    table[i, var_index+j] = True

        np.savetxt(sys.stdout.buffer, table*1, fmt="%.0d") 

class Expr():

    # expr metadata
    def __init__(self, id, index, var_set, expr_set, body):
        self.id = id # name
        self.index = index # position in ordered id list
        self.var_set = var_set # set of vars present in body
        self.expr_set = expr_set # set of other exprs present in body
        self.body = body # body (what comes after =)
    
    def eval(self, truth_val_mapping): # no checks, we take for granted corectness of the expression
        # return truth_val_mapping.values()[0]
        body = self.body
        length = len(body)
        cursor = 0
        stack = [[]]

        while cursor < length:
            token = body[cursor]
            
            if token == '(': # start new recursion level
                stack.append([])

            elif token == ')': # eval recursion level

                result = eval_rec_level(stack[-1], truth_val_mapping)
                
                stack.pop()
                if stack:
                    stack[-1].append(result)
                else:
                    stack.append([result])

            else: # append token to the current recursion level
                stack[-1].append(token)
                
            cursor += 1

        if stack:
            result = eval_rec_level(stack[-1], truth_val_mapping)
        
        return result

def main(arg):
    try:
        tokenizer = Tokenizer(arg)
        tokenizer.run()
        parser = Parser()
        parser.run()
        compiler = Compiler() 
        compiler.run()
    except Exception as error:
        print(f"{arg}: {error}")

if __name__ == '__main__':
    arg = sys.argv[1]
    main(arg)
     