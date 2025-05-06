# %%
import transformers
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# DeBERTa でテキストを読み取って 0-shot で埋込表現を得る

def get_embeddings(texts, model_name='microsoft/deberta-base'):
    # モデルとトークナイザーの読み込み
    model = transformers.AutoModel.from_pretrained(model_name).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    batch_size = 32
    # テキストをトークン化
    embeddings = []
    attention_masks = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors='pt', max_length=128).to("cuda")

        # モデルを評価モードに設定
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            # 出力は二次元で取得、最終的に３次元配列を保存
            last_hidden_states = outputs.last_hidden_state.cpu().numpy()
            embeddings.append(last_hidden_states)
            attention_masks.append(inputs['attention_mask'].cpu().numpy())
            
    # すべてのバッチの埋込表現を結合
    embeddings = np.concatenate(embeddings, axis=0)
    attention_masks = np.concatenate(attention_masks, axis=0)
    print(embeddings.shape)
    print(attention_masks.shape)
    return embeddings, attention_masks

# %%
train_df = pd.read_csv("../dataset/train.csv")
train_texts = train_df['Caption'].tolist()
# 埋込表現を取得
embeddings, attention_masks = get_embeddings(train_texts)
# 埋込表現を保存
np.save("../dataset/deberta_embeddings_train.npy", embeddings)
np.save("../dataset/deberta_attention_masks_train.npy", attention_masks)

test_df = pd.read_csv("../dataset/test.csv")
test_texts = test_df['Caption'].tolist()
# 埋込表現を取得
embeddings, attention_masks = get_embeddings(test_texts)
# 埋込表現を保存
np.save("../dataset/deberta_embeddings_test.npy", embeddings)
np.save("../dataset/deberta_attention_masks_test.npy", attention_masks)


