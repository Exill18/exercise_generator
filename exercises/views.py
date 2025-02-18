# views.py
from django.shortcuts import render
from django import forms
import random
import math
import numpy as np
from functools import lru_cache

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
        "Qual é o número natural z tal que o triplo do número é a soma do seu dobro com menos {constant}?\n"
        "a) Expressão do triplo de z\n"
        "b) Equação matemática\n"
        "c) Resolução e conjunto solução [2.0v]"
    )
    POINTS = 3.5
    IDENTIFIER = "linear_equation"

    @classmethod
    def generate_params(cls):
        return {'constant': random.randint(5, 15)}

    @classmethod
    def get_solution(cls, params):
        return f"3z = 2z - {params['constant']} ⇒ z = -{params['constant']} (Não é natural)"

# 2. Quadratic Equation Exercise
class QuadraticEquationExercise(BaseExercise):
    TEMPLATE = "Resolva a equação quadrática: {a}x² + {b}x + {c} = 0"
    POINTS = 2.5
    IDENTIFIER = "quadratic"

    @classmethod
    def generate_params(cls):
        for _ in range(100):  # Prevent infinite loop
            a = random.choice([1, 1, 1, 2])
            b = random.randint(-5, 5)
            c = random.randint(-10, 10)
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                return {'a': a, 'b': b, 'c': c}
        return {'a': 1, 'b': -2, 'c': 1}  # Fallback

    @classmethod
    def get_solution(cls, params):
        roots = np.roots([params['a'], params['b'], params['c']])
        return [round(float(root), 2) for root in roots]

# 3. SystemOfEquationsExercise
class SystemOfEquationsExercise(BaseExercise):
    TEMPLATE = (
        "Um grupo de {total} pessoas entre H/M/C.\n"
        "H + M = {ratio}× crianças\n"
        "Mulheres = homens + {diff}\n"
        "a) Sistema de equações [0.75v]\n"
        "b) Forma matricial [1.0v]\n"
        "c) Solução [2.5v]"
    )
    POINTS = 4.25
    IDENTIFIER = "system_eq"

    @classmethod
    def generate_params(cls):
        return {
            'total': random.randint(30, 50),
            'ratio': random.randint(2, 5),
            'diff': random.randint(1, 3)
        }

    @classmethod
    def get_solution(cls, params):
        ratio = params['ratio']
        total = params['total']
        diff = params['diff']
        
        c = total / (ratio + 1)
        h = (total - c - diff) / 2
        m = h + diff
        
        return {
            'h': round(h, 2),
            'm': round(m, 2),
            'c': round(c, 2)
        }

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
    POINTS = 2.0
    IDENTIFIER = "matrix_ops"

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
        
        solutions = {}
        
        # Addition
        try:
            solutions['sum'] = (a + b).tolist()
        except ValueError:
            solutions['sum'] = "Impossível - Dimensões diferentes"
        
        # Multiplication
        try:
            solutions['product'] = (a @ b).tolist()
        except ValueError:
            solutions['product'] = "Impossível - Colunas de A ≠ Linhas de B"
        
        return solutions
# 5. TrigonometryExercise
class TrigonometryExercise(BaseExercise):
    TEMPLATE = "Calcule {operation}({angle}°) [1.25v]"
    POINTS = 1.25
    INPUT_TYPE = "number"
    IDENTIFIER = "trig"

    @classmethod
    def generate_params(cls):
        return {
            'angle': random.choice([30, 45, 60, 120, 135, 150]),
            'operation': random.choice(['sin', 'cos', 'tan'])
        }

    @classmethod
    def get_solution(cls, params):
        angle = math.radians(params['angle'])
        return {
            'sin': math.sin(angle),
            'cos': math.cos(angle),
            'tan': math.tan(angle)
        }[params['operation']]

# 6. RationalEquationExercise
class RationalEquationExercise(BaseExercise):
    TEMPLATE = (
        "Determine o maior número real que satisfaz: 'Metade de um número é a soma do seu inverso com três'\n"
        "Equação: (1/2)x = 1/x + 3\n"
        "Solução [1.75v]"
    )
    POINTS = 1.75
    INPUT_TYPE = "number"
    IDENTIFIER = "rational_eq"

    @classmethod
    def generate_params(cls):
        return {'n': random.randint(1, 10), 'c': random.randint(5, 15)}

    @classmethod
    def get_solution(cls, params):
        # Solve (1/2)x = 1/x + 3
        a = 0.5
        b = -3
        c = -1
        discriminant = b**2 - 4*a*c
        return round((-b + math.sqrt(discriminant)) / (2*a), 2)

# 7. EquationVerificationExercise
class EquationVerificationExercise(BaseExercise):
    TEMPLATE = "Prove que -1 não é solução da equação x² - 2x + 3 = 0\nSubstituição [1.25v]"
    POINTS = 1.25
    IDENTIFIER = "eq_verify"

    @classmethod
    def generate_params(cls):
        return {'a': 1, 'b': -2, 'c': 3}

    @classmethod
    def get_solution(cls, params):
        return "(-1)² - 2(-1) + 3 = 1 + 2 + 3 = 6 ≠ 0"

# 8. TrigonometryReferenceExercise
class TrigonometryReferenceExercise(BaseExercise):
    TEMPLATE = "Represente o ângulo de {angle}° no círculo trigonométrico e indique {operation}(π - α) [1.5v]"
    POINTS = 1.5
    INPUT_TYPE = "number"
    IDENTIFIER = "trig_ref"

    @classmethod
    def generate_params(cls):
        return {
            'angle': random.choice([30, 45, 60, 120, 135, 150]),
            'operation': random.choice(['sin', 'cos'])
        }

    @classmethod
    def get_solution(cls, params):
        ref_angle = 180 - params['angle']
        if params['operation'] == 'sin':
            return round(math.sin(math.radians(ref_angle)), 4)
        return round(math.cos(math.radians(ref_angle)), 4)

# 9. MatrixSubtractionExercise (fixed)
class MatrixSubtractionExercise(BaseMatrixExercise):
    TEMPLATE = (
        "Dadas as matrizes:\n"
        "A = {a_str}\n"
        "B = {b_str}\n"
        "Calcule A - B [1.0v]"
    )
    POINTS = 1.0
    IDENTIFIER = "matrix_sub"

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
        "1) y + {coeff}x = -10\n"
        "2) -x + 5y = -9\n"
        "Encontre as coordenadas de interseção [3.75v]"
    )
    POINTS = 3.75
    IDENTIFIER = "substitution_sys"

    @classmethod
    def generate_params(cls):
        return {'coeff': random.randint(2, 5)}

    @classmethod
    def get_solution(cls, params):
        coeff = params['coeff']
        denominator = -1 - 5*coeff
        if denominator == 0:
            return "Sistema impossível"
        x = 41 / denominator
        y = -10 - coeff*x
        return (round(x, 2), round(y, 2))

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
        return [round(det_x/det, 2), round(det_y/det, 2)]

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
        self.fields['answer'] = forms.CharField(
            label=self.exercise_class.render_question(self.params),
            widget=forms.TextInput if exercise_class.INPUT_TYPE == 'text' else forms.NumberInput
        )

def exam_view(request):
    if 'exercises' not in request.session:
        request.session['exercises'] = []

    if request.method == 'POST':
        results = []
        total_score = 0
        
        for idx, ex_data in enumerate(request.session.get('exercises', [])):
            exercise_class = next(
                ex for ex in EXERCISE_TYPES 
                if ex.IDENTIFIER == ex_data['type']
            )
            params = ex_data['params']  # Extract params from ex_data
            form = ExerciseForm(request.POST, prefix=str(idx), exercise_class=exercise_class, params=params)            
            if form.is_valid():
                is_correct = exercise_class.validate_answer(
                    form.cleaned_data['answer'],
                    ex_data['params']
                )
                points = exercise_class.POINTS if is_correct else 0
                total_score += points
                
                results.append({
                    'question': exercise_class.render_question(ex_data['params']),
                    'user_answer': form.cleaned_data['answer'],
                    'correct_answer': exercise_class.get_solution(ex_data['params']),
                    'points': points,
                    'is_correct': is_correct
                })

        # Generate new exam
        new_exercises = []
        for _ in range(10):
            ex_class = random.choice(EXERCISE_TYPES)
            params = ex_class.generate_params()
            new_exercises.append({
                'type': ex_class.IDENTIFIER,
                'params': params
            })
        
        request.session['exercises'] = new_exercises
        return render(request, 'exercises/results.html', {
            'results': results,
            'total_score': round(total_score, 2)
        })

    # Initial GET request
    # Get unique exercise types
    unique_exercises = list(set(EXERCISE_TYPES))
    selected_classes = random.sample(unique_exercises, min(10, len(unique_exercises)))
    
    # Fill remaining slots with random choices if needed
    while len(selected_classes) < 10:
        new_ex = random.choice(EXERCISE_TYPES)
        if new_ex not in selected_classes:
            selected_classes.append(new_ex)
    
    exam_exercises = []
    for ex_class in selected_classes:
        params = ex_class.generate_params()
        exam_exercises.append({
            'type': ex_class.IDENTIFIER,
            'params': params
        })

    # Create forms with actual generated parameters
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