from django.http import JsonResponse

from .privateGPT.privateGPT import generate
from rest_framework.decorators import api_view

from apiDocAssistant.helper import handle_uploaded_file  


@api_view(['GET'])
def get_suggestions_data(request):
    print("request: "+ request.GET.get('query'))
    query = request.GET.get('query')
    model_type = request.GET.get('model_type')
    print("query: "+ query)
    print("model_type: "+ model_type)
    answer = generate(query=query, model_type=model_type)
    return JsonResponse({"message": answer })

@api_view(['POST'])
def upload_file(request):  
    if request.method == 'POST':  
        handle_uploaded_file(request.FILES['file'])  
        return JsonResponse({"message": "File uploaded successfully"})
