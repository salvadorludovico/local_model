from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Caminho para o diretório do modelo
model_path = "./Phi-3.5-mini-instruct"

# Verificando se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Carregando o modelo e o tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Movendo o modelo para GPU (se disponível)
model.to(device)

# Inicializando o FastAPI
app = FastAPI()

# Rota para geração de texto
@app.post("/generate")
async def generate_text(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
