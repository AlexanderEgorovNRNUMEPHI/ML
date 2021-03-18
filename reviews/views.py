from django.shortcuts import render
from .my_function import my_best_function1
from .binary_classification import result_of_bc
from .MC import MC

def index(request):
    if request.method == 'POST':
        my_review = request.POST['text']
        result0 = result_of_bc(my_review)
        result1 = MC(my_review)
        if result0[0] == 0:
            result = "Negative"
        else:
            result = "Positive"
        context = {
            'review_text': my_review,
            'my_best_function0': result,
            'my_best_function1': result1,
        }
        return render(request, 'reviews/index.html', context)
    return render(request, 'reviews/index.html')