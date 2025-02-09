import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    def __init__(self, config_path: str | None = None) -> None:
        if config_path:
            self.config: Dict[str, Any] = self._load_config(config_path)
        else:
            self.config = {}

        self.model: Any = None
        self.tokenizer: Optional[Any] = None
        self.device: str = self.config.get("device", "cpu")
        self.max_length: int = self.config.get("max_length", 512)
        self.temperature: float = self.config.get("temperature", 0.7)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def download_model(self, model_name: str):
        # Download the model from Hugging Face
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token to eos_token if it's not already defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model, self.tokenizer

    def load_model(self):
        # If config specifies 'gguf' as the model type, use llama-cpp
        if self.config.get("model_type") == "gguf":
            from llama_cpp import Llama  # type: ignore
            model_path = self.config["path"]
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.max_length,
                temperature=self.temperature,
                # Additional llama-cpp parameters can be added here
            )
            self.tokenizer = None
            return self.model, None
        else:
            raise NotImplementedError("Only gguf models are supported for now.")
            self.model = AutoModelForCausalLM.from_pretrained()
            self.tokenizer = AutoTokenizer.from_pretrained()
            return self.model, self.tokenizer

    def generate_response(self, prompt: str) -> str:
        if self.config.get("model_type") == "gguf":
            outputs = self.model(
                prompt=prompt,
                max_tokens=self.max_length,
                temperature=self.temperature,
                echo=False
            )
            return outputs["choices"][0]["text"]
        else:
            # Ensure the tokenizer has a pad_token set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Tokenize the input and ensure padding is applied
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate a response with the model
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
