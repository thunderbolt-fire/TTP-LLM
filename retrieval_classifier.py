#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题检索需求分类器
用于判断一个问题是否需要检索/网络搜索的3分类模型
分类标签: 0 = 不需要检索, 1 = 需要一般检索, 2 = 需要深度网络搜索
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import random
import torch

from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

# 设置随机种子以保证结果可重复性
set_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def create_synthetic_dataset():
    """
    创建模拟数据集
    返回一个包含问题和对应标签的 DataFrame
    """
    data = [
        # 不需要检索的问题 (标签 0)
        {"question": "1+1等于多少？", "label": 0},
        {"question": "中国的首都是哪里？", "label": 0},
        {"question": "水的化学式是什么？", "label": 0},
        {"question": "一年有多少天？", "label": 0},
        {"question": "地球的形状是什么？", "label": 0},
        {"question": "哺乳动物的特征是什么？", "label": 0},
        {"question": "三角形的内角和是多少度？", "label": 0},
        {"question": "光速大约是多少？", "label": 0},
        
        # 需要一般检索的问题 (标签 1)
        {"question": "2025年世界人口是多少？", "label": 1},
        {"question": "最新的iPhone型号是什么？", "label": 1},
        {"question": "2024年奥运会在哪里举行？", "label": 1},
        {"question": "当前美元兑换人民币的汇率是多少？", "label": 1},
        {"question": "最近的诺贝尔奖得主是谁？", "label": 1},
        {"question": "最新的科技新闻有哪些？", "label": 1},
        {"question": "2025年电影票房排行榜", "label": 1},
        {"question": "最新的人工智能研究进展", "label": 1},
        
        # 需要深度网络搜索的问题 (标签 2)
        {"question": "如何解决气候变化的具体措施有哪些？", "label": 2},
        {"question": "2025年最新的癌症治疗方法研究", "label": 2},
        {"question": "量子计算在实际应用中的最新突破", "label": 2},
        {"question": "如何实现可持续发展的具体政策建议", "label": 2},
        {"question": "深度学习在医疗诊断中的最新应用案例", "label": 2},
        {"question": "全球能源危机的解决方案分析", "label": 2},
        {"question": "人工智能对未来就业市场的影响研究", "label": 2},
        {"question": "基因编辑技术的伦理问题和监管政策", "label": 2}
    ]
    
    df = pd.DataFrame(data)
    df.to_csv("/workspace/TTP-LLM/retrieval_dataset.csv", index=False)
    print("模拟数据集已保存至 /workspace/TTP-LLM/retrieval_dataset.csv")
    return df

def main():
    # 创建并加载数据集
    create_synthetic_dataset()
    dataset = load_dataset('csv', data_files={'train': '/workspace/TTP-LLM/retrieval_dataset.csv'})
    
    # 标签定义
    labels = ["不需要检索", "需要一般检索", "需要深度网络搜索"]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    num_labels = len(labels)
    
    print("标签列表:", labels)
    print("标签数量:", num_labels)
    
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = model.to("cuda:0")
    
    # 数据预处理函数
    def preprocess_data(examples):
        encoding = tokenizer(examples["question"], padding="max_length", truncation=True, max_length=512)
        encoding["labels"] = examples["label"]
        return encoding
    
    # 预处理数据集
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")
    
    # 训练参数设置
    batch_size = 8
    training_args = TrainingArguments(
        output_dir="/workspace/TTP-LLM/retrieval_classifier_model",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=15,
        weight_decay=0.01,
        no_cuda=False,
        use_cpu=False
    )
    
    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        tokenizer=tokenizer
    )
    
    # 开始训练
    print("开始训练模型...")
    trainer.train()
    
    # 保存模型
    trainer.save_model("/workspace/TTP-LLM/retrieval_classifier_final")
    print("模型已保存至 /workspace/TTP-LLM/retrieval_classifier_final")
    
    # 测试训练后的模型
    test_model()

def test_model():
    """
    测试训练后的模型
    """
    print("\n=== 测试训练后的模型 ===")
    
    # 加载训练好的模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("/workspace/TTP-LLM/retrieval_classifier_final")
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 测试问题
    test_questions = [
        "1+1等于多少？",
        "最新的iPhone型号是什么？",
        "如何解决气候变化的问题？",
        "中国的首都是哪里？",
        "量子计算的最新研究进展？"
    ]
    
    # 进行预测
    for question in test_questions:
        inputs = tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to("cuda:0" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions).item()
        
        # 标签映射
        label_map = {0: "不需要检索", 1: "需要一般检索", 2: "需要深度网络搜索"}
        print("问题:", question)
        print("预测结果:", label_map[predicted_label])
        print("置信度:", "{:.4f}".format(predictions[0][predicted_label].item()))
        print("-" * 50)

if __name__ == "__main__":
    main()
