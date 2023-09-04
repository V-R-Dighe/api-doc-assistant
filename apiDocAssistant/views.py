from django.http import JsonResponse, StreamingHttpResponse
import time

from .privateGPT.privateGPT import generate, generate_data

def get_suggestions(request):
    print("request: "+ request.GET.get('query'))
    query = request.GET.get('query')
    print("query: "+ query)
    result = generate(query=query, mute_stream=False, hide_source=False)
    print("result: "+ result)
    return JsonResponse({"message": result})
