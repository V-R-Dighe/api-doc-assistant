def handle_uploaded_file(f):  
    with open('./source_documents/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)  