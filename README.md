```md
# Exercise Generator

Welcome to the Exercise Generator project! This project is a Django-based web application that 
dynamically generates math exercises for users to solve. It includes various types of exercises
such as linear equations, quadratic equations, systems of equations, matrix operations, trigonometry.

## Features

- Dynamic generation of math exercises
- User-friendly interface for solving exercises
- Automatic validation of user answers
- Display of results and scores
- Integration with MathJax for rendering mathematical expressions

## Project Structure
└── exill18-exercise_generator/
    ├── LICENSE 
    ├── README.md
    ├── manage.py
    ├── exercise_generator/
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── exercises/
        ├── __init__.py
        ├── admin.py
        ├── apps.py
        ├── models.py
        ├── tests.py
        ├── urls.py
        ├── views.py
        ├── migrations/
        │   └── __init__.py
        ├── static/
        │   └── check-for-tex.js
        └── templates/
            └── exercises/
                ├── index.html
                └── results.html

```

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- Django 5.1.2
- NumPy

### Installation

1. Clone the repository:

```sh
git clone https://github.com/Exill18/exercise_generator.git
cd exercise_generator
```

2. Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

4. Apply migrations to set up the database:

```sh
python manage.py migrate
```

5. Run the development server:

```sh
python manage.py runserver
```

6. Open your browser and navigate to `http://127.0.0.1:8000` to access the application.

## Usage

1. On the homepage, you will see a dynamically generated math test with various exercises.
2. Fill in your answers and submit the form.
3. The results page will display your score and the correct answers for each exercise.
4. You can start a new test by clicking the "Novo Teste" button.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Django](https://www.djangoproject.com/)
- [NumPy](https://numpy.org/)
- [MathJax](https://www.mathjax.org/)

