from django.shortcuts import render
from django.http import JsonResponse
from .ai_models.loader import ModelLoader

current_model = None

def chat(request):
    global current_model
    if request.method == "POST":
        model_name = request.POST.get("model")
        if model_name:
        #     current_model = ModelLoader(f"src/chat/ai_models/configs/{model_name}.yaml")
        #     current_model.load_model()
        # else:
            current_model = ModelLoader()
            current_model.download_model(model_name)
    return render(request, "chat/index.html")

def generate_response(request):
    if request.method == "POST":
        user_input = request.POST.get("message")
        if current_model:
            response = current_model.generate_response(user_input)
            return JsonResponse({"response": response})
        return JsonResponse({"error": "No model loaded"}, status=400)