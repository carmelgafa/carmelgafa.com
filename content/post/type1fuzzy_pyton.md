---
title: Fuzzy Inference System implementation in Python
tags: [fuzzy, type1-fuzzy, python]
date: "2020-06-24T14:43:21+02:00"
draft: false
---

This article was first published in [Towards Data Science](https://towardsdatascience.com/fuzzy-inference-system-implementation-in-python-8af88d1f0a6e).

## Introduction

In a [previous article](https://carmelgafa.com/post/item/a-very-brief-introduction-to-fuzzy-logic-and-fuzzy-systems/), we discussed the basics of fuzzy sets and fuzzy inferencing. The report also illustrated the construction of a possible control application using a fuzzy inferencing method. In this article, we will build a multi-input/multi-output fuzzy inference system using the Python programming language. It is assumed that the reader has a clear understanding of fuzzy inferencing and has read the article mentioned previously.

All the code listed in this article is available on [Github](https://github.com/carmelgafa/ml_from_scratch/tree/master/fuzzy_inference).

## System Architecture

The diagram below illustrates the structure of the application. The design is based on several considerations on Fuzzy Inference Systems, some being:

* A Fuzzy Inference System will require input and output variables and a collection of fuzzy rules.

* Both input and output variables will contain a collection of fuzzy sets if the Fuzzy Inference System is of Mamdani type.
* Input and output variables are very similar, but they are used differently by fuzzy rules. During execution, input variables use the input values to the system to fuzzify their sets, that is they determine the degree of belonging of that input value to all of the fuzzy sets of the variable. Each rule contributes to some extent to the output variables; the totality of this contribution will determine the output of the system.

* Fuzzy rules have the structure of the form;

```if {antecedent clauses} then {consequent clauses}```

Therefore a rule will contain several clauses of antecedent type and some clauses of consequent type. Clauses will be of the form:

```{variable name} is {set name}```

![packages](/post/img/art2_fig1.png)

We will discuss some implementation details of the classes developed for this system in the following sections:

### FuzzySet class

A FuzzySet requires the following parameters so that it can be initiated:

* name — the name of the set
* minimum value — the minimum value of the set
* maximum value — the maximum value of the set
* resolution — the number of steps between the minimum and maximum value

It is, therefore, possible to represent a fuzzy set by using two numpy arrays; one that will hold the domain values and one that will hold the degree-of-membership values. Initially, all degree-of-membership values will be all set to zero. It can be argued that if the minimum and maximum values are available together with the resolution of the set, the domain numpy array is not required as the respective values can be calculated. While this is perfectly true, a domain array was preferred in this example project so that the code is more readable and simple.

```python
def create_triangular(cls, name, domain_min, 
    domain_max, res, a, b, c):

  t1fs = cls(name, domain_min, domain_max, res)

  a = t1fs._adjust_domain_val(a)
  b = t1fs._adjust_domain_val(b)
  c = t1fs._adjust_domain_val(c)

  t1fs._dom = np.round(np.maximum(np.minimum
    ((t1fs._domain-a)/(b-a), (c-t1fs._domain)/(c-b))
    , 0), t1fs._precision)
  ```

In the context of a fuzzy variable, all the sets will have the same minimum, maximum and resolution values.

As we are dealing with a discretized domain, it will be necessary to adjust any value used to set or retrieve the degree-of-membership to the closest value in the domain array.

```python
def _adjust_domain_val(self, x_val):
  return self._domain[np.abs(
      self._domain-x_val).argmin()]
```

The class contains methods whereby a set of a given shape can be constructed given a corresponding number of parameters. In the case of a triangular set, for example, three parameters are provided, two that define the extents of the sets and one for the apex. It is possible to construct a triangular set by using these three parameters as can be seen in the figure below.

![set creation](/post/img/art2_eqn1.jpg)

Since the sets are based on numpy arrays, the equation above can be translated directly to code, as can be seen below. Sets having different shapes can be constructed using a similar method.

```python
def create_triangular(cls, name, domain_min, 
    domain_max, res, a, b, c):

  t1fs = cls(name, domain_min, domain_max, res)

  a = t1fs._adjust_domain_val(a)
  b = t1fs._adjust_domain_val(b)
  c = t1fs._adjust_domain_val(c)

  t1fs._dom = np.round(np.maximum(np.minimum(
      (t1fs._domain-a)/(b-a), 
      (c-t1fs._domain)/(c-b)), 0),
       t1fs._precision)
```

The _FuzzySet_ class also contains union, intersection and negation operators that are necessary so that inferencing can take place. All operator methods return a new fuzzy set with the result of the operation that took place.

```python
def union(self, f_set):

    result = FuzzySet(
        f'({self._name}) union ({f_set._name})', 
        self._domain_min, 
        self._domain_max, self._res)

    result._dom = np.maximum(self._dom, f_set._dom)

    return result
```

Finally, we implemented the ability to obtain a crisp result from a fuzzy set using the centre-of-gravity method that is referred to in some detail in the previous article. It is important to mention that there is a large number of defuzzification methods are available in the literature. Still, as the centre-of-gravity method is overwhelmingly popular, it is used in this implementation.

```python
def cog_defuzzify(self):

  num = np.sum(
      np.multiply(self._dom, self._domain))
  
  den = np.sum(self._dom)

  return num/den
```

### Fuzzy Variable classes

![variable classes](/post/img/art2_fig2.jpg)

As discussed previously, variables can be of input or output in type, with the difference affecting the fuzzy inference calculation. A FuzzyVariable is a collection of sets that are held in a python dictionary having the set name as the key. Methods are available to add FuzzySets to the variable, where such sets will take the variable’s limits and resolution.

For input variables, fuzzification is carried out by retrieving the degree-of-membership of all the sets in the variable for a given domain value. The degree-of-membership is stored in the set as it will be required by the rules when they are evaluated.

```python
def fuzzify(self, val):

    # get dom for each set and store it -
    # it will be required for each rule
    for set_name, f_set in self._sets.items():
        f_set.last_dom_value = f_set[val]
```

Output variables will ultimately produce the result of a fuzzy inference iteration. This means that for Mamdani-type systems, as we are building here, output variables will hold the union of the fuzzy contributions from all the rules, and will subsequently defuzzify this result to obtain a crisp value that can be used in real-life applications.

Therefore, output variables will require an additional FuzzySet attribute that will hold the output distribution for that variable, where the contribution that was resulting from each rule and added using the set union operator. The defuzzification result can then be obtained by calling the centre-of-gravity method for output distribution set.

```python
class FuzzyOutputVariable(FuzzyVariable):

    def __init__(self, name, min_val, max_val, res):
        super().__init__(name, min_val, 
            max_val, res)
        self._output_distribution = 
            FuzzySet(name, min_val, max_val, res)

    def add_rule_contribution(self, 
        rule_consequence):
        
        self._output_distribution = 
            self._output_distribution.union(
                rule_consequence)

    def get_crisp_output(self):
        return 
           self._output_distribution.cog_defuzzify()
```

### Fuzzy Rules classes

The FuzzyClause class requires two attributes; a fuzzy variable and a fuzzy set so that the statement

```variable is set```

can be created. Clauses are used to implement statements that can be chained together to form the antecedent and consequent parts of the rule.
When used as an antecedent clause, the FuzzyClause returns the last degree-of-membership value of the set, that is calculated during the fuzzification stage as we have seen previously.

The rule will combine the degree-of-membership values from the various antecedent clauses using the min operator, obtaining the rule activation that is then used in conjunction with the consequent clauses to obtain the contribution of the rule to the output variables. This operation is a two-step process:

* The activation value is combined with the consequent FuzzySet using the min operator, that will act as a threshold to the degree-of-membership values of the FuzzySet.
* The resultant FuzzySet is combined with the FuzzySets obtained from the other rules using the union operator, obtaining the output distribution for that variable.

```python
# execution methods for a FuzzyClause
# that contains a FuzzyVariable;
#  _variable
# and a FuzzySet; _set
  
def evaluate_antecedent(self):
    return self._set.last_dom_value

def evaluate_consequent(self, activation):
    self._variable.add_rule_contribution(
        self._set.min_scalar(activation))
```

The FuzzyRule class will, therefore, require two attributes:

* a list containing the antecedent clauses and
* a list containing the consequent clauses

During the execution of the FuzzyRule, the procedure explained above is carried out. The FuzzyRule coordinates all the tasks by utilizing all the various FuzzyClauses as appropriate.


```python
def evaluate(self):
    # rule activation initialize to
    # 1 as min operator will be performed
    rule_activation = 1

    # execute all antecedent clauses,
    # keeping the minimum of the returned
    # doms to determine the activation
    for ante_clause in self._antecedent:
        rule_activation =
            min(ante_clause.evaluate_antecedent(),
            rule_activation)

    # execute consequent clauses, each output 
    # variable will update its 
    # output_distribution set
    for consequent_clause in self._consequent:
        consequent_clause.evaluate_consequent(
            rule_activation)
```

### Fuzzy System Class — Bringing it all together.

At the topmost level of this architecture, we have the FuzzySystem that coordinates all activities between the FuzzyVariables and FuzzyRules. Hence the system contains the input and output variables, that are stored in python dictionaries using variable-names as keys and a list of the rules.

One of the challenges presented at this stage is the method that the end-user will use to add rules, that should ideally abstract the implementation detail of the FuzzyClause classes. The method that was implemented consists of providing two python dictionaries that will contain the antecedent and consequent clauses of the rule in the following format;

```variable name : set name```

A more user-friendly method is to provide the rule as a string and then parse that string to create the rule, but this seemed an unnecessary overhead for a demonstration application.

```python
    def add_rule(self, antecedent_clauses, consequent_clauses):
        '''
        adds a new rule to the system.
        Arguments:
        -----------
        antecedent_clauses -- dict,
            {variable_name:set_name, ...}
        consequent_clauses -- dict,
            {variable_name:set_name, ...}
        '''
        # create a new rule
        new_rule = FuzzyRule()

        for var_name, set_name in
            antecedent_clauses.items():

            # get variable by name
            var = self.get_input_variable(var_name)

            # get set by name

            f_set = var.get_set(set_name)

            # add clause
            new_rule.add_antecedent_clause(
                var, f_set)

        for var_name, set_name in
            consequent_clauses.items():

            var = self.get_output_variable(var_name)

            f_set = var.get_set(set_name)

            new_rule.add_consequent_clause(
                var, f_set)

        # add the new rule
        self._rules.append(new_rule)
```

Addition of a new rule to the FuzzySystem

The execution of the inference process can be achieved with a few lines of code given this structure, where the following steps are carried out;

1. The output distribution sets of all the output variables are cleared.
2. The input values to the system are passed to the corresponding input variables so that each set in the variable can determine its degree-of-membership for that input value.
3. Execution of the Fuzzy Rules takes place, meaning that the output distribution sets of all the output variables will now contain the union of the contributions from each rule.

```python
  # clear the fuzzy consequences
  # as we are evaluating a new set of inputs.
  # can be optimized by comparing if the inputs
  # have changes from the previous
  # iteration.
  self._clear_output_distributions()

  # Fuzzify the inputs. The degree of membership
  # will be stored in each set

  for input_name, input_value in 
    input_values.items():
    
    self._input_variables[input_name].fuzzify(
        input_value)

  # evaluate rules
  for rule in self._rules:
    rule.evaluate()

  # finally, defuzzify all output distributions 
  # to get the crisp outputs
  
  output = {}
  for output_var_name, output_var in 
    self._output_variables.items():

    output[output_var_name] =
        output_var.get_crisp_output()

  return output
```

As a final note, the Fuzzy Inferencing System implemented here contains additional functions to plot fuzzy sets and variables and to obtain information about an inference step execution.

## Library Use Example

In this section, we will discuss the use of the fuzzy inference system. In particular, we will implement the fan speed case study that was designed in the previous article in this series.

A fuzzy system begins with the consideration of the input and output variables, and the design of the fuzzy sets to explain that variable.

The variables will require a lower and upper limit and, as we will be dealing with discrete fuzzy sets, the resolution of the system. Therefore a variable definition will look as follows

```temp = FuzzyInputVariable('Temperature', 10, 40, 100)```

where the variable ‘Temperature’ ranges between 10 and 40 degrees and is discretized in 100 bins.

The fuzzy sets define for the variable will require different parameters depending on their shape. In the case of triangular sets, for example, three parameters are needed, two for the lower and upper extremes having a degree of membership of 0 and one for the apex which has a degree-of-membership of 1. A triangular set definition for variable ‘Temperature’ can, therefore, look as follows;

```temp.add_triangular('Cold', 10, 10, 25)```

where the set called ‘Cold’ has extremes at 10 and 25 and apex at 10 degrees. In our system, we considered two input variables, ‘Temperature’ and ‘Humidity’ and a single output variable ‘Speed’. Each variable us described by three fuzzy sets. The definition of the output variable ‘Speed’ looks as follows:

```python
motor_speed = FuzzyOutputVariable('Speed', 
    0, 100, 100)
motor_speed.add_triangular('Slow', 0, 0, 50)
motor_speed.add_triangular('Moderate', 10, 50, 90)
motor_speed.add_triangular('Fast', 50, 100, 100)
```

As we have seen before, the fuzzy system is the entity that will contain these variables and fuzzy rules. Hence the variables will have to be added to a system as follows:

```python
system = FuzzySystem() 
system.add_input_variable(temp) 
system.add_input_variable(humidity)
system.add_output_variable(motor_speed)
```

## Fuzzy Rules
A fuzzy system executes fuzzy rules to operate of the form

```If x1 is S and x2 is M then y is S```

where the If part of the rule contains several antecedent clauses and the then section will include several consequent clauses. To keep things simple, we will assume rules that require an antecedent clause from each input variable and are only linked together with an ‘and’ statement. It is possible to have statements linked by ‘or’ and statements can also contain operators on the sets like ‘not’.

The simplest way to add a fuzzy rule to our system is to provide a list of the antecedent clauses and consequent clauses. One method of doing so is by using a python dictionary that contains

```Variable:Set```
entries for the clause sets. Hence the above rule can be implemented as follows:

```python
system.add_rule(  
{ 'Temperature':'Cold',
'Humidity':'Wet' },
{ 'Speed':'Slow'})
```

Execution of the system involves inputting values for all the input variables and getting the values for the output values in return. Again this is achieved through the use of dictionaries that us the name of the variables as keys.

```python
output = system.evaluate_output({
'Temperature':18,
'Humidity':60  })
```

The system will return a dictionary containing the name of the output variables as keys, and the defuzzified result as values.

## Conclusion

In this article, we have looked at the practical implementation of a fuzzy inference system. Whilst the library presented here will require some further work so that it can be used in real projects, including validation and exception handling, it can serve as the basis for projects that require Fuzzy Inferencing. It is also recommended to look at some open-source projects that are available, in particular skfuzzy, a fuzzy logic toolbox for SciPy.
In the next article, we will examine ways whereby a fuzzy system can be created from a dataset so that that fuzzy logic can be used in machine learning scenarios. Similarly to this introduction to Fuzzy Logic concepts, a practical article will follow.