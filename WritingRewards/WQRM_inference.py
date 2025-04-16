import os, torch, json
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from huggingface_hub import hf_hub_download

def load_data(filename: str) -> List[Dict[str, Any]]:
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def create_dataloader(filename: str, batch_size: int) -> DataLoader:
    data = load_data(filename)
    
    examples = []
    for d in data:
        paragraph1, paragraph2 = "", "None"
        label_cls, label_reg = -1, -1
        if d["sample_type"] == "pairwise":
            label_cls = 0 if d["reference_preference"] == "1" else 1
            paragraph1 = d["paragraph1"]
            paragraph2 = d["paragraph2"]
        else:
            paragraph1 = d["paragraph"]
            label_reg = d["zscore"]

        rationale = "" if "rationale" not in d else d["rationale"]
            
        examples.append({
            "sample_type": d["sample_type"], 
            "paragraph1": paragraph1, 
            "paragraph2": paragraph2, 
            "label_cls": label_cls, 
            "label_reg": label_reg,
            "rationale": rationale
        })
    
    def collate_fn(batch):
        paragraphs1 = [x['paragraph1'] for x in batch]
        paragraphs2 = [x['paragraph2'] for x in batch]
        rationales = [x['rationale'] for x in batch]

        labels_cls = torch.LongTensor([x['label_cls'] for x in batch])
        labels_reg = torch.FloatTensor([x['label_reg'] for x in batch])
        
        return paragraphs1, paragraphs2, rationales, labels_cls, labels_reg
    
    return DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class MBertWQRM(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and NLU model for both local and online sources
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlu = AutoModel.from_pretrained(model_name)
        hidden_size = self.nlu.config.hidden_size
        
        # Create regression head
        self.regression_head = self._create_regression_head(hidden_size)
        
        # Initialize weights
        self._init_weights(self.regression_head)
        
        # Try to load the custom heads from the model
        self._load_custom_heads(model_name)
        
        self.regression_scale = 10.0  # Scale factor for regression output
        self.to(self.device)

    def _load_custom_heads(self, model_name: str):
        """Load custom heads from local or online models"""
        # Check if model_name is a local path or an online model
        if os.path.exists(model_name):
            # Local path
            heads_path = os.path.join(model_name, 'heads.pth')
            if os.path.exists(heads_path):
                heads_state = torch.load(heads_path, map_location=self.device)
                self.regression_head.load_state_dict(heads_state['regression_head'])
        else:
            try:
                # Use huggingface_hub to download with built-in caching
                repo_id = model_name
                filename = "heads.pth"
                
                # Download the file (will use cache if available)
                cached_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=None,  # Use default HF cache directory
                    resume_download=True
                )
                
                # Load the heads from the cached path
                if os.path.exists(cached_file):
                    heads_state = torch.load(cached_file, map_location=self.device)
                    self.regression_head.load_state_dict(heads_state['regression_head'])
                else:
                    print(f"Warning: Could not find custom heads at {cached_file}")
            except Exception as e:
                print(f"Error loading custom heads: {e}")

    def reload_weights(self, path):
        """DEPRECATED: Use load_model instead.
        Loads only the regression head weights."""
        import warnings
        warnings.warn("reload_weights is deprecated; use load_model instead", DeprecationWarning)
        self.regression_head.load_state_dict(torch.load(path))

    def _create_regression_head(self, hidden_size):
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, paragraphs1, paragraphs2, rationales=None):
        N = len(paragraphs1)
        if rationales is None:
            rationales = [""] * N

        SEP_TOKEN_ID = self.tokenizer.sep_token_id
        
        # Process each pair of paragraphs
        all_input_ids, all_attention_masks, all_p1_ranges, all_p2_ranges = [], [], [], []
        
        for p1, p2, r in zip(paragraphs1, paragraphs2, rationales):
            p1_tokens = self.tokenizer(p1, add_special_tokens=False)['input_ids']
            p2_tokens = self.tokenizer(p2, add_special_tokens=False)['input_ids']

            r_tokens = []
            if r != "":
                r_tokens = self.tokenizer(r, add_special_tokens=False)['input_ids']

            
            input_ids = p1_tokens + [SEP_TOKEN_ID] + p2_tokens
            if len(r_tokens) > 0:
                input_ids += [SEP_TOKEN_ID] + r_tokens

            attention_mask = [1] * len(input_ids)
            
            p1_range = [0, len(p1_tokens)]
            p2_range = [len(p1_tokens) + 1, len(p1_tokens) + len(p2_tokens) + 1]
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_p1_ranges.append(p1_range)
            all_p2_ranges.append(p2_range)
        
        # Pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            padding_length = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_masks.append(attention_mask + [0] * padding_length)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_input_ids).to(self.device)
        attention_mask = torch.tensor(padded_attention_masks).to(self.device)
        p1_ranges = torch.tensor(all_p1_ranges).to(self.device)
        p2_ranges = torch.tensor(all_p2_ranges).to(self.device)
        
        # Process through model
        outputs = self.nlu(input_ids=input_ids, attention_mask=attention_mask)

        p1_outputs, p2_outputs = [], []
        for i in range(N):
            p1_outputs.append(outputs.last_hidden_state[i, p1_ranges[i][0]:p1_ranges[i][1], :].mean(dim=0).unsqueeze(0))
            p2_outputs.append(outputs.last_hidden_state[i, p2_ranges[i][0]:p2_ranges[i][1], :].mean(dim=0).unsqueeze(0))

        p1_outputs = torch.cat(p1_outputs, dim=0)
        p2_outputs = torch.cat(p2_outputs, dim=0)

        reg_logits = self.regression_head(p1_outputs) * self.regression_scale  # Scale to 0-10 range
        cls_logits = torch.nn.functional.cosine_similarity(p1_outputs, p2_outputs, dim=-1)
        cls_logits = torch.clamp(cls_logits, min=1e-7, max=1)

        return cls_logits, reg_logits
    
    def predict_pair(self, paragraph1: str, paragraph2: str) -> str:
        self.eval()
        with torch.no_grad():
            cls_logits, _ = self([paragraph1], [paragraph2])
            pred = (cls_logits <= 0.5).long()
            return "1" if pred.item() == 0 else "2"
        
    def predict_regression(self, paragraph: str) -> float:
        self.eval()
        with torch.no_grad():
            _, reg_logits = self([paragraph], ["None"])
            return reg_logits.item()
    
    def evaluate(self, val_loader):
        self.eval()
        total_loss_cls, total_loss_reg = 0, 0
        total_acc, total_mse, total_mae = 0, 0, 0
        N_cls, N_reg = 0, 0

        all_reg_logits = []
        all_reg_labels = []
        with torch.no_grad():
            for paragraphs1, paragraphs2, rationales, labels_cls, labels_reg in val_loader:
                outputs = self(paragraphs1, paragraphs2, rationales)
                cls_logits, reg_logits = outputs
                
                for label_cls, label_reg, cls_logit, reg_logit in zip(labels_cls, labels_reg, cls_logits, reg_logits):
                    label_cls = label_cls.to(cls_logit.device)
                    label_reg = label_reg.to(reg_logit.device)
                    if label_cls != -1:
                        N_cls += 1
                        total_loss_cls += -torch.log(cls_logit) if label_cls == 0 else -torch.log(1 - cls_logit)

                        preds = (cls_logit <= 0.5).long()
                        total_acc += (preds == label_cls).sum().item()
                    else:
                        N_reg += 1
                        diff = (reg_logit - label_reg)
                        total_loss_reg += diff.pow(2).mean()
                        total_mse += diff.pow(2).mean()
                        total_mae += diff.abs().mean()
                        all_reg_logits.append(reg_logit.item())
                        all_reg_labels.append(label_reg.item())

        total_loss_cls = total_loss_cls / (N_cls + 1e-8)
        total_loss_reg = (total_loss_reg / (N_reg + 1e-8)) / 4.0
        total_loss = total_loss_cls + total_loss_reg
        total_acc = total_acc / (N_cls + 1e-8)
        total_mse = total_mse / (N_reg + 1e-8)
        total_mae = total_mae / (N_reg + 1e-8)

        val_corr = 0
        if N_reg > 0:
            val_corr = np.corrcoef(all_reg_logits, all_reg_labels)[0, 1]

        return {"loss_total": total_loss, "loss_cls": total_loss_cls, "loss_reg": total_loss_reg, "acc": total_acc, "mse": total_mse, "mae": total_mae, "N_cls": N_cls, "N_reg": N_reg, "val_corr": val_corr}

    def save_model(self, save_dir):
        """Save the model to a directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the base model and tokenizer
        self.nlu.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save the custom heads
        heads_state = {
            'regression_head': self.regression_head.state_dict()
        }
        torch.save(heads_state, os.path.join(save_dir, 'heads.pth')) 
    
    @classmethod
    def load_model(cls, model_path):
        model = cls(model_path)
        return model

if __name__ == "__main__":
    # Short snippet from the New Yorker
    paragraph1 = """"Those people seem so pleased to see each other," Manager said. And I realized she'd been watching them as closely as I had. "Yes, they seem so happy," I said. "But it's strange because they also seem upset." "Oh, Klara," Manager said quietly. "You never miss a thing, do you?" Then Manager was silent for a long time, holding her sign in her hand and staring across the street, even after the pair had gone out of sight. Finally she said, "Perhaps they hadn't met for a long time. A long, long time. Perhaps when they last held each other like that, they were still young." "Do you mean, Manager, that they lost each other?" She was quiet for another moment. "Yes," she said, eventually. "That must be it. They lost each other. And perhaps just now, just by chance, they found each other again." Manager's voice wasn't like her usual one, and though her eyes were on the outside, I thought she was now looking at nothing in particular. I even started to wonder what passers-by would think to see Manager herself in the window with us for so long."""

    # Generated by GPT-4 on a similar plot
    paragraph2 = """From my position by the store window, I watched intently as the two figures outside displayed what seemed to be conflicting emotions, their faces caught between joy and distress at their encounter. Manager stood beside me, her usual businesslike demeanor softening as she observed my careful attention to the scene unfolding before us. "You notice everything, don't you, Klara?" she remarked, her voice carrying a hint of admiration. The pair outside continued their awkward dance of emotions, their body language speaking volumes about their complicated history. After they departed, Manager's gaze lingered on the now-empty sidewalk, and she mused that they must have been separated for years before this chance meeting. Her voice took on an unusually contemplative tone, quite different from her typical practical manner, and I noticed how her eyes seemed to look beyond the street, perhaps into her own past. The moment revealed not just the complexity of human reunions but also how such encounters could stir deep memories in unexpected observers, making me wonder about Manager's own story of lost connections."""


    # model_folder = "models/WQRM/" # Change this to wherever the folder is
    # model_folder = "models/WQRM-PRE/" # Change this to wherever the folder is

    model_folder = "Salesforce/WQRM-PRE"

    model = MBertWQRM(model_folder)

    print(f"Pair-preference prediction (1 is New Yorker, 2 is GPT-4o): {model.predict_pair(paragraph1, paragraph2)}")

    print("WQRM Score of each paragraph")
    print("Paragraph 1: ", model.predict_regression(paragraph1))
    print("Paragraph 2: ", model.predict_regression(paragraph2))
