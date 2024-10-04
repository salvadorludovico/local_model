from fastapi import FastAPI
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# Caminho para o diretório do modelo ONNX
model_path = "../Phi-3-mini-4k-instruct-onnx"

# Carregando o tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Inicializando o ONNX Runtime
session = ort.InferenceSession(f"{model_path}/model.onnx", providers=['CPUExecutionProvider'])

# Inicializando o FastAPI
app = FastAPI()

# Função auxiliar para preparar os inputs no formato necessário para ONNX Runtime
def prepare_inputs(input_text):
    # Tokenizando a entrada e convertendo para tensores
    inputs = tokenizer(input_text, return_tensors="np")
    # Preparando os inputs para o ONNX (numpy arrays)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Rota para geração de texto
@app.post("/generate")
async def generate_text(input_text: str):
    # Preparando inputs para ONNX
    inputs_onnx = prepare_inputs(input_text)
    
    # Executando o modelo ONNX
    outputs = session.run(None, inputs_onnx)
    
    # Decodificando a sequência gerada
    generated_ids = outputs[0]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return {"generated_text": generated_text[len(input_text):].strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
