import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# Dataset Preparation
class GSM8kDataset(Dataset):
    def __init__(self, tokenizer, stage, c, max_length):
        '''
            stage: 去除前stage个reasoning steps
        '''
        ds = load_dataset("openai/gsm8k", "main")
        self.data = []
        for ex in ds['train']:
            question = ex['question']
            
            # answer#### 后的部分
            answer = ex['answer'].split('#### ')[1].strip()
            
            # steps为所有<< >>内的部分，使用正则表达式
            import re
            steps = re.findall(r"<<(.*?)>>", ex['answer'])

            self.data.append((question, steps, answer))
        self.tokenizer = tokenizer
        self.stage = stage
        self.latent_steps = stage * c
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, reasoning_steps, answer = self.data[idx]
        # print(f"Question: {question}\nReasoning Steps: {reasoning_steps}\nAnswer: {answer}")

        stage_reasoning_steps = reasoning_steps[self.stage : ] if self.stage < len(reasoning_steps) else []

        before_bot = f"Question: {question}\n\nReasoning:<bot>"
        after_eot = f"<eot>{stage_reasoning_steps}\n\nAnswer: {answer}"

        before_bot_text = self.tokenizer(
            before_bot,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )
        after_eot_text = self.tokenizer(
            after_eot,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )
        # print(before_bot_text)
        before_bot_position_ids = torch.cumsum(before_bot_text.attention_mask, dim=1) - 1
        after_eot_position_ids = before_bot_position_ids[:, -1] + self.latent_steps + torch.cumsum(after_eot_text.attention_mask, dim=1)

        after_eot_attention_mask = torch.cat([
                                                before_bot_text.attention_mask, 
                                                torch.ones((before_bot_text.attention_mask.size(0), self.latent_steps), device=before_bot_text.attention_mask.device),
                                                after_eot_text.attention_mask
                                            ], dim=1)
        return {
            "before_bot": before_bot_text.input_ids.squeeze(0),
            "after_eot": after_eot_text.input_ids.squeeze(0),
            "before_bot_attention": before_bot_text.attention_mask.squeeze(0),
            "after_eot_attention": after_eot_attention_mask.squeeze(0),
            "before_bot_position_ids": before_bot_position_ids.squeeze(0),
            "after_eot_position_ids": after_eot_position_ids.squeeze(0),
        }

# Define the Model Wrapper with Latent Mode
class CoconutModel(nn.Module):
    def __init__(self, base_model_name):
        super(CoconutModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map=device, torch_dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, position_ids=None, latent_steps=0):
        outputs = self.model.model(
                               input_ids, 
                               attention_mask=attention_mask, 
                               use_cache=True,
                               position_ids=position_ids,
                               output_hidden_states=True,
                            )
        kv_cache = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        next_position_ids = position_ids[:, -1:]
        next_mask = attention_mask[:, :]
        for _ in range(latent_steps):
            next_input = hidden_states[:, -1, :].unsqueeze(1)  # Use last hidden state as input embedding
            next_position_ids = next_position_ids + 1
            next_mask = torch.cat([next_mask, torch.ones((next_mask.size(0), 1), device=next_mask.device)], dim=1)
            outputs = self.model.model(
                                    inputs_embeds=next_input, 
                                    past_key_values=kv_cache, 
                                    use_cache=True,
                                    position_ids=next_position_ids,
                                    output_hidden_states=True,
                                )
            hidden_states = outputs.last_hidden_state
            kv_cache = outputs.past_key_values

        return hidden_states, kv_cache

# Multi-Stage Training with Gradual Replacement of Reasoning Steps
def train_multistage_model(model, stages, c=1):
    for stage in stages:
        dataset = GSM8kDataset(tokenizer, stage['stage'], c, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(stage['epochs']):
            total_loss = 0
            for batch in tqdm(dataloader):
                before_bot = batch["before_bot"].to(device)
                after_eot = batch["after_eot"].to(device)
                before_bot_attention = batch["before_bot_attention"].to(device)
                after_eot_attention = batch["after_eot_attention"].to(device)
                before_bot_position_ids = batch["before_bot_position_ids"].to(device)
                after_eot_position_ids = batch["after_eot_position_ids"].to(device)
                optimizer.zero_grad()

                # Calculate latent steps based on current stage
                total_steps = before_bot.size(1)  # Total token length
                latent_steps = stage['stage'] * c

                # Perform n + 1 forward passes for latent thoughts
                hidden_states, kv_cache = model(before_bot, before_bot_attention, position_ids=before_bot_position_ids, latent_steps=latent_steps)

                # Use KV cache to save computation for the language loss
                outputs = model.model(
                    input_ids=after_eot,
                    attention_mask=after_eot_attention,
                    position_ids=after_eot_position_ids,
                    labels=after_eot,
                    past_key_values=kv_cache,
                )
                language_loss = outputs.loss

                # Combine latent and language losses (mask questions and latent thoughts)
                total_loss_value = language_loss.mean()
                total_loss_value.backward()
                optimizer.step()

                total_loss += total_loss_value.item()

            print(f"Stage {stage['stage']}, Epoch {epoch}/{stage['epochs']}, Loss: {total_loss / len(dataloader):.4f}")



if __name__ == "__main__":
    # Hyperparameters
    base_model_name = "Qwen/Qwen2-0.5B"
    max_length = 256
    batch_size = 1
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<bot>", "<eot>"]})

    stages = [
        {"stage": 0, "epochs": 6},
        {"stage": 1, "epochs": 3},
        {"stage": 2, "epochs": 3},
    ]
    model = CoconutModel(base_model_name)
    model.to(device)
    train_multistage_model(model, stages, c=1)