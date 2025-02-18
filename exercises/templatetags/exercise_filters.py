# filepath: /c:/Users/franc/code/matematica/exercise_generator/exercises/templatetags/exercise_filters.py
from django import template

register = template.Library()

@register.filter
def format_dict(template_str, params):
    return template_str.format(**params)