# views.py
from django.shortcuts import render
from django import forms
import random
import math
import numpy as np
from functools import lru_cache
from fractions import Fraction

class BaseExercise:
    TEMPLATE = ""
    INPUT_TYPE = "text"
    POINTS = 0
    IDENTIFIER = "base"

    @classmethod
    def generate_params(cls):
        raise NotImplementedError

    @classmethod
    def get_solution(cls, params):
        raise NotImplementedError

    @classmethod
    def validate_answer(cls, user_input, params):
        solution = cls.get_solution(params)
        try:
            if isinstance(solution, (int, float)):
                return abs(float(user_input) - solution) < 0.01
            elif isinstance(solution, list):
                user_values = [float(x.strip()) for x in user_input.split(",")]
                return all(abs(u - s) < 0.01 for u, s in zip(user_values, solution))
            return user_input.strip() == str(solution)
        except:
            return False

    @classmethod
    def render_question(cls, params):
        return cls.TEMPLATE.format(**params)
    
#base class for matrix exercises
class BaseMatrixExercise(BaseExercise):
    MAX_DIM = 3
    INPUT_TYPE = "text"
    
    @classmethod
    def generate_matrix(cls, min_dim=1):
        rows = random.randint(min_dim, cls.MAX_DIM)
        cols = random.randint(min_dim, cls.MAX_DIM)
        return [
            [random.randint(1, 9) for _ in range(cols)]
            for _ in range(rows)
        ]
    
    @classmethod
    def matrix_to_str(cls, matrix):
        if len(matrix) == 1:
            return f"[{', '.join(map(str, matrix[0]))}]"
        return '[\n' + '\n'.join(
            f"  [{', '.join(map(str, row))}]" 
            for row in matrix
        ) + '\n]'

# 1. Linear Equation Exercise
class LinearEquationExercise(BaseExercise):
    TEMPLATE = (
        "Qual é o número natural z tal que o {multiplos} do número é a soma do seu {multiplos2} com menos {constant}?\n"
        "a) Expressão do {multiplos} de z [0.75v]\n"
        "b) Equação matemática [1.0v]\n"
        "c) Resolução e conjunto solução [2.0v]"
    )

    # lista com possiveis multiplicacoes do numero
    multiplos = ["triplo", "dobro", "quadruplo"]
    POINTS = 3.75  # Sum of parts
    IDENTIFIER = "linear_equation"
    PARTS = [
        ('a', 0.75, "Expressão do {multiplos} de z"),
        ('b', 1.0, "Equação matemática"),
        ('c', 2.0, "Resolução e conjunto solução")
    ]

    @classmethod
    def generate_params(cls):
        selected_multiplos = random.choice(cls.multiplos)
        multiplier = {'dobro': 2, 'triplo': 3, 'quadruplo': 4}[selected_multiplos]
        constant = random.randint(1, 10)
        while constant >= multiplier:
            constant = random.randint(1, 10)
        remaining_multiplos = [m for m in cls.multiplos if m != selected_multiplos]
        selected_multiplos2 = random.choice(remaining_multiplos)
        return {'constant': constant, 'multiplos': selected_multiplos, 'multiplos2': selected_multiplos2}

    @classmethod
    def get_parts(cls, params):
        return [
            ('a', 0.75, "Expressão do {multiplos} de z"),
            ('b', 1.0, "Equação matemática"),
            ('c', 2.0, "Resolução e conjunto solução")
        ]

    @classmethod
    def get_solution(cls, params):
        constant = params['constant']
        multiplier = {'dobro': 2, 'triplo': 3, 'quadruplo': 4}[params['multiplos']]
        z = Fraction(constant, multiplier - 1)
        z_str = f"{z.numerator}/{z.denominator}" if z.denominator != 1 else f"{z.numerator}"
        return {
            'a': {'display': f'{multiplier}z', 'value': f'{multiplier}z'},
            'b': {'display': f'{multiplier}z = z + {constant}', 'value': f'{multiplier}z = z + {constant}'},
            'c': {'display': f'z = {z_str}', 'value': z}
        }
# 2. Quadratic Equation Exercise
class QuadraticEquationExercise(BaseExercise):
    TEMPLATE = "Resolva a equação quadrática: {a}x² + {b}x + {c} = 0"
    PARTS = [('answer', 2.5, "Solução")]
    IDENTIFIER = "quadratic"
    INPUT_TYPE = "text"

    @classmethod
    def generate_params(cls):
        while True:
            a = random.choice([1, 2, 3, 4, 5])
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            d = b**2 - 4*a*c
            if d >= 0 and math.isqrt(d)**2 == d:
                return {'a': a, 'b': b, 'c': c}

    @classmethod
    def get_solution(cls, params):
        a, b, c = params['a'], params['b'], params['c']
        d = b**2 - 4*a*c
        sqrt_d = math.isqrt(d)
        x1 = Fraction(-b + sqrt_d, 2*a)
        x2 = Fraction(-b - sqrt_d, 2*a)
        x1_str = f"{x1.numerator}/{x1.denominator}" if x1.denominator !=1 else f"{x1.numerator}"
        x2_str = f"{x2.numerator}/{x2.denominator}" if x2.denominator !=1 else f"{x2.numerator}"
        return {'answer': {'display': f"x₁ = {x1_str}, x₂ = {x2_str}", 'value': [x1, x2]}}

# 3. SystemOfEquationsExercise
class SystemOfEquationsExercise(BaseExercise):
    TEMPLATE = (
        "Um grupo de {total} pessoas entre Homens (H), Mulheres (M) e Crianças (C).\n"
        "Sabe-se que:\n"
        "1. O número de Homens e Mulheres juntos é o {ratio} do número de Crianças\n"
        "2. O número de Mulheres é {diff} a mais que o número de Homens\n"
        "a) Sistema de equações [0.75v]\n"
        "b) Forma matricial [1.0v]\n"
        "c) Valores exatos de H/M/C [2.5v]"
    )
    PARTS = [
        ('a', 0.75, "Sistema de equações"),
        ('b', 1.0, "Forma matricial"),
        ('c', 2.5, "Solução")
    ]
    POINTS = 4.25
    IDENTIFIER = "system_eq"
    RATIO_MAP = {2: 'dobro', 3: 'triplo', 4: 'quádruplo', 5: 'quíntuplo'}

    @classmethod
    def generate_params(cls):
        while True:
            ratio_num = random.choice(list(cls.RATIO_MAP.keys()))
            diff = random.randint(1, 5)
            
            # Find valid k where total = k*(ratio_num + 1)
            min_k = max(1, math.ceil(10/(ratio_num + 1)))
            max_k = math.floor(50/(ratio_num + 1))
            
            for k in random.sample(range(min_k, max_k + 1), max_k - min_k + 1):
                total = k * (ratio_num + 1)
                C = k
                numerator = ratio_num * C - diff
                
                # Ensure H is positive integer
                if numerator > 0 and numerator % 2 == 0:
                    H = numerator // 2
                    M = H + diff
                    
                    if H > 0 and M > 0 and (H + M + C) == total:
                        return {
                            'total': total,
                            'ratio': cls.RATIO_MAP[ratio_num],
                            'ratio_num': ratio_num,
                            'diff': diff,
                            '_solution': (H, M, C)
                        }

    @classmethod
    def get_solution(cls, params):
        H, M, C = params['_solution']
        return {
            'a': {
                'display': (
                    f"1) H + M = {params['ratio']}C\n"
                    f"2) M = H + {params['diff']}\n"
                    f"3) H + M + C = {params['total']}"
                ),
                'value': [
                    f"H + M = {params['ratio']}C",
                    f"M = H + {params['diff']}",
                    f"H + M + C = {params['total']}"
                ]
            },
            'b': {
                'display': (
                    "Matriz de coeficientes:\n"
                    f"{BaseMatrixExercise.matrix_to_str([[1,1,-params['ratio_num']], [-1,1,0], [1,1,1]])}\n"
                    "Matriz de constantes:\n"
                    f"{BaseMatrixExercise.matrix_to_str([[0], [params['diff']]], [params['total']])}"
                )
            },
            'c': {
                'display': f"H = {H}, M = {M}, C = {C}",
                'value': (H, M, C)
            }
        }

    @classmethod
    def validate_answer(cls, user_input, params, part):
        solution = cls.get_solution(params)
        if part == 'c':
            try:
                user_h = int(user_input.split('H=')[1].split(',')[0])
                user_m = int(user_input.split('M=')[1].split(',')[0])
                user_c = int(user_input.split('C=')[1])
                return (user_h, user_m, user_c) == solution['c']['value']
            except:
                return False
        # Validation for parts a/b remains the same
        return super().validate_answer(user_input, params, part)
    
    @classmethod
    def render_question(cls, params):
        return cls.TEMPLATE.format(**params)
    

# 4. MatrixOperationsExercise (fixed)
class MatrixOperationsExercise(BaseMatrixExercise):
    TEMPLATE = (
        "Dadas as matrizes:\n"
        "A = {a_str}\n"
        "B = {b_str}\n"
        "Calcule:\n"
        "a) A + B [1.0v]\n"
        "b) A × B [1.0v]"
    )
    PARTS = [('a', 1.0, "A + B"), ('b', 1.0, "A × B")]
    POINTS = 2.0
    IDENTIFIER = "matrix_ops"

    @classmethod
    def generate_params(cls):
        a = cls.generate_matrix()
        b = cls.generate_matrix(min_dim=len(a))
        return {
            'a': a,
            'b': b,
            'a_str': cls.matrix_to_str(a),
            'b_str': cls.matrix_to_str(b)
        }
    


    @classmethod
    def get_solution(cls, params):
        a = np.array(params['a'])
        b = np.array(params['b'])
        solutions = {}
        
        # Addition
        try:
            sum_matrix = (a + b).tolist()
            solutions['a'] = {'display': "A + B = " + cls.matrix_to_str(sum_matrix)}
        except ValueError:
            solutions['a'] = {'display': "A soma não é possível devido a dimensões diferentes."}
        
        # Multiplication
        try:
            product_matrix = (a @ b).tolist()
            solutions['b'] = {'display': "A × B = " + cls.matrix_to_str(product_matrix)}
        except ValueError:
            solutions['b'] = {'display': "A multiplicação não é possível devido a dimensões incompatíveis."}
        
        return solutions
# 5. TrigonometryExercise
class TrigonometryExercise(BaseExercise):
    TEMPLATE = "Calcule {operation}({angle}°)"
    PARTS = [('answer', 1.25, "Resposta")]
    IDENTIFIER = "trig"
    INPUT_TYPE = "text"

    @classmethod
    def generate_params(cls):
        return {'angle': random.choice([30, 45, 60]), 'operation': random.choice(['sin', 'cos', 'tan'])}
    

    @classmethod
    def validate_single_answer(cls, user_input, solution_part):
        symbols = {
            '√3/2': ['√3/2', 'raiz3/2', 'sqrt(3)/2'],
            '1/2': ['1/2', '0.5'],
            '√3': ['√3', 'raiz3', 'sqrt(3)']
        }
        solution = solution_part['display'].lower()
        user_clean = user_input.strip().lower()
        for sym in symbols.get(solution, [solution]):
            if user_clean == sym.lower():
                return True
        try:
            return abs(float(user_input) - float(solution_part['value'])) < 0.01
        except:
            return False

    @classmethod
    def get_solution(cls, params):
        angle = params['angle']
        operation = params['operation']
        ref = {
            (60, 'sin'): ('√3/2', math.sqrt(3)/2),
            (30, 'sin'): ('1/2', 0.5),
            (45, 'sin'): ('√2/2', math.sqrt(2)/2),
            (60, 'cos'): ('1/2', 0.5),
            (30, 'cos'): ('√3/2', math.sqrt(3)/2),
            (45, 'cos'): ('√2/2', math.sqrt(2)/2),
            (45, 'tan'): ('1', 1.0),
            (60, 'tan'): ('√3', math.sqrt(3)),
            (30, 'tan'): ('1/√3', 1/math.sqrt(3)),
        }
        display, value = ref.get((angle, operation), (f"{operation}({angle}°)", None))
        return {'answer': {'value': value, 'display': display}}
# 6. RationalEquationExercise
class RationalEquationExercise(BaseExercise):
    TEMPLATE = (
        "Determine o maior número real que satisfaz:\n"
        "{a_texto} de um número é igual a {b} dividido pelo número mais {c}\n"
        "Equação: {equation}\n"
        "Solução [1.75v]"
    )
    POINTS = 1.75
    INPUT_TYPE = "text"
    IDENTIFIER = "rational_eq"
    PARTS = [('answer', 1.75, "Solução")]
    
    MULTIPLIER_MAP = {
        2: "O dobro",
        3: "O triplo",
        0.5: "Metade"
    }

    @classmethod
    def generate_params(cls):
        while True:
            # Generate equation parameters that guarantee exact solution
            a = random.choice([0.5, 2, 3])
            p = random.randint(1, 5)
            q = random.choice([2, 3, 5, 6, 7, 8, 10, 11, 12])  # Non-perfect squares
            
            c = 2 * a * p
            b = a * (q - p**2)
            
            # Ensure natural language consistency and valid equation
            if b != 0 and (c**2 + 4*a*b) > 0:
                return {
                    'a': a,
                    'b': b,
                    'c': c,
                    'p': p,
                    'q': q,
                    'a_texto': cls.MULTIPLIER_MAP[a],
                    'equation': f"({a})x = {b}/x + {c}"
                }

    @classmethod
    def get_solution(cls, params):
        exact_form = f"{params['p']} + √{params['q']}"
        numerical_value = params['p'] + math.sqrt(params['q'])
        return {
            'answer': {
                'display': exact_form,
                'value': numerical_value,
                'exact_form': exact_form
            }
        }

    @classmethod
    def validate_answer(cls, user_input, params):
        solution = cls.get_solution(params)['answer']
        user_clean = user_input.strip().lower()
        
        # Accept both forms: 3 + √11 or 3+sqrt(11)
        user_clean = user_clean.replace(" ", "").replace("√", "sqrt").replace("raiz", "sqrt")
        solution_clean = solution['exact_form'].lower().replace(" ", "").replace("√", "sqrt")
        
        # Check exact form match
        if user_clean == solution_clean:
            return True
        
        # Check numerical approximation
        try:
            return abs(float(user_input) - solution['value']) < 0.01
        except:
            return False
        
# 7. EquationVerificationExercise
class EquationVerificationExercise(BaseExercise):
    TEMPLATE_NOT_SOLUTION = "Prove que {value} não é solução da equação {a}x² + {b}x + {c} = 0\nSubstituição [1.25v]"
    TEMPLATE_IS_SOLUTION = "Prove que {value} é solução da equação {a}x² + {b}x + {c} = 0\nSubstituição [1.25v]"
    POINTS = 1.25
    IDENTIFIER = "eq_verify"
    PARTS = [('answer', 1.25, "Substituição e verificação")]

    @classmethod
    def generate_params(cls):
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        is_solution = random.choice([True, False])

        # Generate a value that is a solution or not a solution to the equation
        value = random.randint(-10, 10)
        if is_solution:
            # Ensure the value is a solution
            value = cls.find_solution(a, b, c)
        else:
            # Ensure the value is not a solution
            while a * value**2 + b * value + c == 0:
                value = random.randint(-10, 10)

        return {'a': a, 'b': b, 'c': c, 'value': value, 'is_solution': is_solution}

    @staticmethod
    def find_solution(a, b, c):
        # Find a value that is a solution to the equation a*x^2 + b*x + c = 0
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            sqrt_d = math.isqrt(discriminant)
            if sqrt_d**2 == discriminant:
                x1 = (-b + sqrt_d) // (2*a)
                x2 = (-b - sqrt_d) // (2*a)
                return random.choice([x1, x2])
        return 0  # Default value if no solution is found

    @classmethod
    def get_solution(cls, params):
        a = params['a']
        b = params['b']
        c = params['c']
        value = params['value']
        is_solution = params['is_solution']
        result = a * value**2 + b * value + c
        if is_solution:
            return {'answer': {'display': f"{a}({value})² + {b}({value}) + {c} = {result} = 0", 'value': result}}
        else:
            return {'answer': {'display': f"{a}({value})² + {b}({value}) + {c} = {result} ≠ 0", 'value': result}}

    @classmethod
    def render_question(cls, params):
        if params['is_solution']:
            return cls.TEMPLATE_IS_SOLUTION.format(**params)
        else:
            return cls.TEMPLATE_NOT_SOLUTION.format(**params)
        
# 8. TrigonometryReferenceExercise
class TrigonometryReferenceExercise(BaseExercise):
    TEMPLATE = "Represente o ângulo de {angle}° no círculo trigonométrico e indique {operation}(π - α) [1.5v]"
    POINTS = 1.5
    INPUT_TYPE = "number"
    IDENTIFIER = "trig_ref"
    PARTS = [('answer', 1.5, "Resposta")]

    @classmethod
    def generate_params(cls):
        return {
            'angle': random.choice([30, 45, 60, 120, 135, 150]),
            'operation': random.choice(['sin', 'cos'])
        }

    @classmethod
    def get_solution(cls, params):
        angle = 180 - params['angle']
        operation = params['operation']
        exact_values = {
            (30, 'sin'): '1/2', (45, 'sin'): '√2/2', (60, 'sin'): '√3/2',
            (30, 'cos'): '√3/2', (45, 'cos'): '√2/2', (60, 'cos'): '1/2',
            (120, 'sin'): '√3/2', (120, 'cos'): '-1/2',
            (135, 'sin'): '√2/2', (135, 'cos'): '-√2/2',
            (150, 'sin'): '1/2', (150, 'cos'): '-√3/2',
        }
        display = exact_values.get((angle, operation), f"{operation}({angle}°)")
        return {'answer': {'display': display}}

# 9. MatrixSubtractionExercise (fixed)
class MatrixSubtractionExercise(BaseMatrixExercise):
    TEMPLATE = (
        "Resolva a subtração das matrizes:\n"
        "A = {a_str}\n"
        "B = {b_str}\n"
        "Calcule A - B [1.0v]"
    )
    POINTS = 1.0
    IDENTIFIER = "matrix_sub"
    PARTS = [('answer', 1.0, "A - B")]

    @classmethod
    def generate_params(cls):
        a = cls.generate_matrix()
        b = cls.generate_matrix()
        return {
            'a': a,
            'b': b,
            'a_str': cls.matrix_to_str(a),
            'b_str': cls.matrix_to_str(b)
        }

    @classmethod
    def get_solution(cls, params):
        a = np.array(params['a'])
        b = np.array(params['b'])
        
        try:
            result = (a - b).tolist()
        except ValueError:
            result = "Impossível - Dimensões diferentes"
        
        return result
# 10. SubstitutionSystemExercise
class SubstitutionSystemExercise(BaseExercise):
    TEMPLATE = (
        "Sistema de equações:\n"
        "1) y + {coeff1}x = {constant1}\n"
        "2) {coeff2}x + {coeff3}y = {constant2}\n"
        "Encontre as coordenadas de interseção [3.75v]"
    )
    POINTS = 3.75
    IDENTIFIER = "substitution_sys"
    PARTS = [('answer', 3.75, "Coordenadas de interseção")]

    @classmethod
    def generate_params(cls):
        coeff1 = random.randint(1, 5)
        coeff2 = random.randint(-5, 5)
        coeff3 = random.randint(1, 5)
        constant1 = random.randint(-20, 20)
        constant2 = random.randint(-20, 20)
        return {
            'coeff1': coeff1,
            'coeff2': coeff2,
            'coeff3': coeff3,
            'constant1': constant1,
            'constant2': constant2
        }

    @classmethod
    def get_solution(cls, params):
        coeff1 = params['coeff1']
        coeff2 = params['coeff2']
        coeff3 = params['coeff3']
        constant1 = params['constant1']
        constant2 = params['constant2']

        # Solve the system of equations using substitution
        # Equation 1: y + coeff1 * x = constant1
        # Equation 2: coeff2 * x + coeff3 * y = constant2

        # Solve for y in terms of x from Equation 1
        # y = constant1 - coeff1 * x
        # Substitute into Equation 2
        # coeff2 * x + coeff3 * (constant1 - coeff1 * x) = constant2
        # coeff2 * x + coeff3 * constant1 - coeff3 * coeff1 * x = constant2
        # (coeff2 - coeff3 * coeff1) * x = constant2 - coeff3 * constant1
        denom = coeff2 - coeff3 * coeff1
        if denom == 0:
            return "Sistema impossível"
        x = Fraction(constant2 - coeff3 * constant1, denom)
        y = constant1 - coeff1 * x
        x_str = f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"
        y_str = f"{y.numerator}/{y.denominator}" if y.denominator != 1 else f"{y.numerator}"
        return {'answer': {'display': f"x = {x_str}, y = {y_str}"}}
# 11. CramersRuleExercise

class CramersRuleExercise(BaseExercise):
    TEMPLATE = (
        "Resolva o sistema usando Regra de Cramer:\n"
        "{a}x + {b}y = {c}\n"
        "{d}x + {e}y = {f}\n"
        "[2.5v]"
    )
    POINTS = 2.5
    IDENTIFIER = "cramer"
    PARTS = [('answer', 2.5, "Solução")]

    @classmethod
    def generate_params(cls):
        while True:
            a = random.randint(1, 5)
            b = random.randint(1, 5)
            d = random.randint(2, 6)
            e = random.randint(0, 4)
            det = a*e - b*d
            if det != 0:  # Ensure system has unique solution
                return {
                    'a': a, 'b': b, 'c': random.randint(1, 10),
                    'd': d, 'e': e, 'f': random.randint(3, 12)
                }

    @classmethod
    def get_solution(cls, params):
        a, b, c, d, e, f = params.values()
        det = a*e - b*d
        det_x = c*e - b*f
        det_y = a*f - c*d
        x = Fraction(det_x, det)
        y = Fraction(det_y, det)
        x_str = f"{x.numerator}/{x.denominator}" if x.denominator !=1 else f"{x.numerator}"
        y_str = f"{y.numerator}/{y.denominator}" if y.denominator !=1 else f"{y.numerator}"
        return f"x = {x_str}, y = {y_str}"
    

EXERCISE_TYPES = [
    LinearEquationExercise,
    QuadraticEquationExercise,
    SystemOfEquationsExercise,
    MatrixOperationsExercise,
    TrigonometryExercise,
    RationalEquationExercise,
    EquationVerificationExercise,
    TrigonometryReferenceExercise,
    MatrixSubtractionExercise,
    SubstitutionSystemExercise,
    CramersRuleExercise
]

class ExerciseForm(forms.Form):
    def __init__(self, *args, exercise_class=None, params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exercise_class = exercise_class
        self.params = params
        for part in exercise_class.PARTS:
            label = part[2].format(**params)
            self.fields[f'answer_{part[0]}'] = forms.CharField(label=label, required=True)

def exam_view(request):
    if 'exercises' not in request.session:
        request.session['exercises'] = []

    if request.method == 'POST':
        results = []
        total_score = 0
        exercises = request.session.get('exercises', [])
        
        for idx, ex_data in enumerate(exercises):
            ex_class = next(e for e in EXERCISE_TYPES if e.IDENTIFIER == ex_data['type'])
            form = ExerciseForm(request.POST, prefix=str(idx), 
                              exercise_class=ex_class, params=ex_data['params'])
            if form.is_valid():
                solution = ex_class.get_solution(ex_data['params'])
                part_scores = {}
                for part in ex_class.PARTS:
                    part_id = part[0]
                    user_ans = form.cleaned_data.get(f'answer_{part_id}', '')
                    sol_part = solution.get(part_id, {})
                    is_correct = ex_class.validate_single_answer(user_ans, sol_part)
                    part_scores[part_id] = is_correct
                    total_score += part[1] if is_correct else 0
                results.append({
                    'question': ex_class.render_question(ex_data['params']),
                    'parts': [{'user': form.cleaned_data.get(f'answer_{p[0]}', ''),
                              'correct': solution[p[0]]['display'],
                              'points': p[1]} for p in ex_class.PARTS]
                })
        
        # Generate new exam
        new_exercises = []
        for _ in range(10):
            ex_class = random.choice(EXERCISE_TYPES)
            params = ex_class.generate_params()
            new_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})
        
        request.session['exercises'] = new_exercises
        return render(request, 'exercises/results.html', {
            'results': results,
            'total_score': round(total_score, 2)
        })

    # Initial GET request
    unique_exercises = list(set(EXERCISE_TYPES))
    selected_classes = random.sample(unique_exercises, min(10, len(unique_exercises)))
    
    while len(selected_classes) < 10:
        new_ex = random.choice(EXERCISE_TYPES)
        if new_ex not in selected_classes:
            selected_classes.append(new_ex)
    
    exam_exercises = []
    for ex_class in selected_classes:
        params = ex_class.generate_params()
        exam_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})

    forms = []
    questions = []
    for idx, exercise_data in enumerate(exam_exercises):
        ex_class = next(ex for ex in EXERCISE_TYPES if ex.IDENTIFIER == exercise_data['type'])
        forms.append(ExerciseForm(
            exercise_class=ex_class,
            params=exercise_data['params'],
            prefix=str(idx)
        ))
        questions.append(ex_class.render_question(exercise_data['params']))

    request.session['exercises'] = exam_exercises
    form_question_pairs = zip(forms, questions)
    return render(request, 'exercises/index.html', {'form_question_pairs': form_question_pairs})
    if 'exercises' not in request.session:
        request.session['exercises'] = []

    if request.method == 'POST':
        results = []
        total_score = 0
        exercises = request.session.get('exercises', [])
        
        for idx, ex_data in enumerate(exercises):
            ex_class = next(e for e in EXERCISE_TYPES if e.IDENTIFIER == ex_data['type'])
            form = ExerciseForm(request.POST, prefix=str(idx), 
                              exercise_class=ex_class, params=ex_data['params'])
            if form.is_valid():
                solution = ex_class.get_solution(ex_data['params'])
                part_scores = {}
                for part in ex_class.PARTS:
                    part_id = part[0]
                    user_ans = form.cleaned_data.get(f'answer_{part_id}', '')
                    sol_part = solution.get(part_id, {})
                    is_correct = ex_class.validate_single_answer(user_ans, sol_part)
                    part_scores[part_id] = is_correct
                    total_score += part[1] if is_correct else 0
                results.append({
                    'question': ex_class.render_question(ex_data['params']),
                    'parts': [{'user': form.cleaned_data.get(f'answer_{p[0]}', ''),
                              'correct': solution[p[0]]['display'],
                              'points': p[1]} for p in ex_class.PARTS]
                })
        
        # Generate new exam
        new_exercises = []
        for _ in range(10):
            ex_class = random.choice(EXERCISE_TYPES)
            params = ex_class.generate_params()
            new_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})
        
        request.session['exercises'] = new_exercises
        return render(request, 'exercises/results.html', {
            'results': results,
            'total_score': round(total_score, 2)
        })

    # Initial GET request
    unique_exercises = list(set(EXERCISE_TYPES))
    selected_classes = random.sample(unique_exercises, min(10, len(unique_exercises)))
    
    while len(selected_classes) < 10:
        new_ex = random.choice(EXERCISE_TYPES)
        if new_ex not in selected_classes:
            selected_classes.append(new_ex)
    
    exam_exercises = []
    for ex_class in selected_classes:
        params = ex_class.generate_params()
        exam_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})

    forms = []
    questions = []
    for idx, exercise_data in enumerate(exam_exercises):
        ex_class = next(ex for ex in EXERCISE_TYPES if ex.IDENTIFIER == exercise_data['type'])
        forms.append(ExerciseForm(
            exercise_class=ex_class,
            params=exercise_data['params'],
            prefix=str(idx)
        ))
        questions.append(ex_class.render_question(exercise_data['params']))

    request.session['exercises'] = exam_exercises
    return render(request, 'exercises/index.html', {'forms': forms, 'questions': questions})
    if 'exercises' not in request.session:
        request.session['exercises'] = []

    if request.method == 'POST':
        results = []
        total_score = 0
        exercises = request.session.get('exercises', [])
        
        for idx, ex_data in enumerate(exercises):
            ex_class = next(e for e in EXERCISE_TYPES if e.IDENTIFIER == ex_data['type'])
            form = ExerciseForm(request.POST, prefix=str(idx), 
                              exercise_class=ex_class, params=ex_data['params'])
            if form.is_valid():
                solution = ex_class.get_solution(ex_data['params'])
                part_scores = {}
                for part in ex_class.PARTS:
                    part_id = part[0]
                    user_ans = form.cleaned_data.get(f'answer_{part_id}', '')
                    sol_part = solution.get(part_id, {})
                    is_correct = ex_class.validate_single_answer(user_ans, sol_part)
                    part_scores[part_id] = is_correct
                    total_score += part[1] if is_correct else 0
                results.append({
                    'question': ex_class.render_question(ex_data['params']),
                    'parts': [{'user': form.cleaned_data.get(f'answer_{p[0]}', ''),
                              'correct': solution[p[0]]['display'],
                              'points': p[1]} for p in ex_class.PARTS]
                })
        
        # Generate new exam
        new_exercises = []
        for _ in range(10):
            ex_class = random.choice(EXERCISE_TYPES)
            params = ex_class.generate_params()
            new_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})
        
        request.session['exercises'] = new_exercises
        return render(request, 'exercises/results.html', {
            'results': results,
            'total_score': round(total_score, 2)
        })

    # Initial GET request
    unique_exercises = list(set(EXERCISE_TYPES))
    selected_classes = random.sample(unique_exercises, min(10, len(unique_exercises)))
    
    while len(selected_classes) < 10:
        new_ex = random.choice(EXERCISE_TYPES)
        if new_ex not in selected_classes:
            selected_classes.append(new_ex)
    
    exam_exercises = []
    for ex_class in selected_classes:
        params = ex_class.generate_params()
        exam_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})

    forms = []
    for idx, exercise_data in enumerate(exam_exercises):
        ex_class = next(ex for ex in EXERCISE_TYPES if ex.IDENTIFIER == exercise_data['type'])
        forms.append(ExerciseForm(
            exercise_class=ex_class,
            params=exercise_data['params'],
            prefix=str(idx)
        ))

    request.session['exercises'] = exam_exercises
    return render(request, 'exercises/index.html', {'forms': forms})
    if 'exercises' not in request.session:
        request.session['exercises'] = []

    if request.method == 'POST':
        results = []
        total_score = 0
        exercises = request.session.get('exercises', [])
        
        for idx, ex_data in enumerate(exercises):
            ex_class = next(e for e in EXERCISE_TYPES if e.IDENTIFIER == ex_data['type'])
            form = ExerciseForm(request.POST, prefix=str(idx), 
                              exercise_class=ex_class, params=ex_data['params'])
            if form.is_valid():
                solution = ex_class.get_solution(ex_data['params'])
                part_scores = {}
                for part in ex_class.PARTS:
                    part_id = part[0]
                    user_ans = form.cleaned_data.get(f'answer_{part_id}', '')
                    sol_part = solution.get(part_id, {})
                    is_correct = ex_class.validate_single_answer(user_ans, sol_part)
                    part_scores[part_id] = is_correct
                    total_score += part[1] if is_correct else 0
                results.append({
                    'question': ex_class.render_question(ex_data['params']),
                    'parts': [{'user': form.cleaned_data.get(f'answer_{p[0]}', ''),
                              'correct': solution[p[0]]['display'],
                              'points': p[1]} for p in ex_class.PARTS]
                })
        
        # Generate new exam
        new_exercises = []
        for _ in range(10):
            ex_class = random.choice(EXERCISE_TYPES)
            params = ex_class.generate_params()
            new_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})
        
        request.session['exercises'] = new_exercises
        return render(request, 'exercises/results.html', {
            'results': results,
            'total_score': round(total_score, 2)
        })

    # Initial GET request
    unique_exercises = list(set(EXERCISE_TYPES))
    selected_classes = random.sample(unique_exercises, min(10, len(unique_exercises)))
    
    while len(selected_classes) < 10:
        new_ex = random.choice(EXERCISE_TYPES)
        if new_ex not in selected_classes:
            selected_classes.append(new_ex)
    
    exam_exercises = []
    for ex_class in selected_classes:
        params = ex_class.generate_params()
        exam_exercises.append({'type': ex_class.IDENTIFIER, 'params': params})

    forms = []
    for idx, exercise_data in enumerate(exam_exercises):
        ex_class = next(ex for ex in EXERCISE_TYPES if ex.IDENTIFIER == exercise_data['type'])
        forms.append(ExerciseForm(
            exercise_class=ex_class,
            params=exercise_data['params'],
            prefix=str(idx)
        ))

    request.session['exercises'] = exam_exercises
    return render(request, 'exercises/index.html', {'forms': forms})