from fastapi import FastAPI
from transformers import AutoTokenizer
import onnxruntime as ort

# Caminho para o diretório do modelo ONNX
model_path = "../Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"

# Carregando o tokenizador
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct-onnx")

# Carregando o modelo ONNX com o onnxruntime
ort_session = ort.InferenceSession(model_path)

# Inicializando o FastAPI
app = FastAPI()

# Função para preparar a entrada do modelo
def prepare_input(text):
    inputs = tokenizer(text, return_tensors="np")
    return inputs['input_ids'], inputs['attention_mask']

# Função para fazer a inferência com o ONNX
def generate_onnx(input_ids, attention_mask):
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs

# Rota para geração de texto
@app.post("/generate")
async def generate_text(input_text: str):
    input_ids, attention_mask = prepare_input(input_text)
    outputs = generate_onnx(input_ids, attention_mask)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
