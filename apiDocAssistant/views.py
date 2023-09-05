from django.http import JsonResponse, StreamingHttpResponse
import time

from .privateGPT.privateGPT import generate, generate_data
from rest_framework.decorators import api_view

from apiDocAssistant.helper import handle_uploaded_file  


@api_view(['GET'])
def get_suggestions(request):
    print("request: "+ request.GET.get('query'))
    query = request.GET.get('query')
    print("query: "+ query)
    result = generate(query=query, hide_source=False)
    print("result: "+ result)
    return JsonResponse({"message": result})

@api_view(['GET'])
def get_suggestions_data(request):
    query = request.GET.get('query')

    def generate_stream():
        yield f"Starting data generation for query: {query}\n"

        # Generate data and stream it
        for chunk in generate_data(query=query, hide_source=False):
            print("chunk: "+ chunk)
            yield chunk

    response = StreamingHttpResponse(generate_stream(), content_type="text/plain")
    response['Content-Disposition'] = f'attachment; filename="suggestions_data.txt"'
    return response

@api_view(['POST'])
def upload_file(request):  
    if request.method == 'POST':  
        handle_uploaded_file(request.FILES['file'])  
        return JsonResponse({"message": "File uploaded successfully"})
