import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Импортируем модель (предполагаем, что ViT находится в vit_model.py)
# Если модель в этом же файле, импорт не нужен
from model import ViT 

def evaluate_model(model_path, test_dir, image_size=224, batch_size=32, num_classes=1000):
    """
    Функция для оценки модели на тестовом датасете.
    
    Args:
        model_path (str): Путь к весам модели (.pth или .pt)
        test_dir (str): Путь к папке с тестовыми данными
        image_size (int): Размер изображения
        batch_size (int): Размер батча
        num_classes (int): Количество классов
    """
    
    # 1. Настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # 2. Загрузка модели
    print(f"Загрузка модели из {model_path}...")
    model = ViT(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.0  # На инференсе dropout отключаем
    )
    
    # Загрузка весов
    checkpoint = torch.load(model_path, map_location=device)
    # Если чекпоинт содержит словарь с ключом 'state_dict'
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()  # Режим оценки (отключает Dropout и BatchNorm)
    print("Модель загружена и переключена в режим eval().")
    
    # 3. Подготовка данных
    # Трансформы для тестирования (без аугментаций!)
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Загрузка тестовых данных из {test_dir}...")
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    print(f"Найдено классов: {len(class_names)}")
    print(f"Классы: {class_names}")
    
    # 4. Сбор предсказаний и истинных меток
    all_preds = []
    all_labels = []
    
    print("Начало оценки...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Прямой проход
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Обработано {batch_idx + 1}/{len(test_loader)} батчей")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 5. Расчет метрик
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*50)
    
    # Общая точность (Accuracy)
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nОбщая точность (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision, Recall, F1 (Macro - среднее арифметическое по классам)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro):    {recall_macro:.4f}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")
    
    # Precision, Recall, F1 (Weighted - с учетом размера класса)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\nPrecision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted):    {recall_weighted:.4f}")
    print(f"F1-Score (Weighted):  {f1_weighted:.4f}")
    
    # Подробный отчет по каждому классу
    print("\n" + "="*50)
    print("ОТЧЕТ ПО КЛАССАМ")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # 6. Матрица ошибок (Confusion Matrix)
    print("Построение матрицы ошибок...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Нормализация для лучшего отображения (в процентах)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Сохранение матрицы
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Нормализованная Матрица Ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Матрица ошибок сохранена в 'confusion_matrix.png'")
    plt.close()
    
    # 7. Сохранение результатов в текстовый файл
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ\n")
        f.write("="*50 + "\n\n")
        f.write(f"Общая точность (Accuracy): {accuracy:.4f}\n")
        f.write(f"Precision (Macro): {precision_macro:.4f}\n")
        f.write(f"Recall (Macro): {recall_macro:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n\n")
        f.write("Отчет по классам:\n")
        f.write(str(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)))
    
    print("\nРезультаты сохранены в 'evaluation_results.txt'")
    print("="*50)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    # === НАСТРОЙКИ ===
    MODEL_PATH = 'vit_best.pth'        # Путь к файлу весов модели
    TEST_DIR = 'data/test'             # Путь к папке с тестовыми изображениями
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 10                   # Укажите ваше количество классов
    
    # Проверка существования файлов
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл модели не найден: {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(TEST_DIR):
        print(f"Ошибка: Папка с данными не найдена: {TEST_DIR}")
        exit(1)
    
    # Запуск оценки
    results = evaluate_model(
        model_path=MODEL_PATH,
        test_dir=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )