import customtkinter as ctk
import tkinter as tk
import pandas as pd
from rapidfuzz import process

# Загрузка датасета один раз
dataset = pd.read_csv('id-name.csv')

# Приведение к единому формату
dataset['name_normalized'] = dataset['name'].str.lower().str.strip()

def get_name_by_appid(appid):
    """Получение имени игры по AppID"""
    row = dataset.loc[dataset['AppID'] == appid]
    if not row.empty:
        return row.iloc[0]['name']
    return None

def get_appid_by_name(name):
    """Получение AppID игры по имени (с нечётким поиском)"""
    name_normalized = name.lower().strip()
    match = process.extractOne(name_normalized, dataset['name_normalized'])
    if match and match[1] > 80:  # Порог точности (например, 80%)
        row = dataset.loc[dataset['name_normalized'] == match[0]]
        if not row.empty:
            return row.iloc[0]['AppID']
    return None

# Загрузка данных
file_path = 'games_ym_raz.csv'
games = pd.read_csv(file_path)
# Загрузка модели
import joblib
knn_model = joblib.load('nearest_neighbors_model.pkl')
game_ids = games['AppID']
features = games.drop(columns=['AppID'])
# Функция для рекомендации
def recommend_games_v1(game_id, data, model, n_recommendations=10):
    # Получаем фичи игры по ID
    game_features = data[data['AppID'] == game_id].drop(columns=['AppID'])
    if game_features.empty:
        return f"Игра с ID {game_id} не найдена!"

    # Находим соседей
    distances, indices = model.kneighbors(game_features, n_neighbors=n_recommendations+1)

    # Возвращаем рекомендованные ID игр
    recommended_ids = data.iloc[indices[0]]['AppID'].values[1:]  # Пропускаем саму игру
    return recommended_ids

# Настройка темы и начальных параметров CustomTkinter
ctk.set_appearance_mode("dark")  # Вариант оформления: "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Цветовая тема: "blue", "dark-blue", "green"

# Создаем главное окно
root = ctk.CTk()
root.title("Рекомендация игр")
root.geometry("550x380")
root.resizable(False, False)

# Функция обработки введенной строки
def process_input():
    user_input = input_entry.get()  # Получаем текст из поля ввода
    if user_input.strip():  # Проверяем, что строка не пустая
        # Обрабатываем строку (переворачиваем)
        game_id_to_recommend = get_appid_by_name(user_input)
        recommendations = recommend_games_v1(game_id_to_recommend, games, knn_model)
        results=[]
        for i in recommendations:
            results=results+[get_name_by_appid(i)]
        # Очищаем поле ввода
        input_entry.delete(0, tk.END)

        # Выводим обработанную строку и обновляем список
        processed_label.configure(text=f"Обработанная строка: {get_name_by_appid(game_id_to_recommend)}")
        results_text.configure(state="normal")  # Разрешаем запись
        results_text.delete(1.0, "end")  # Очищаем текстовое поле
        results_text.insert("end", "\n".join(results))  # Выводим список
        results_text.configure(state="disabled")  # Запрещаем редактирование

# Создаем элементы интерфейса
input_label = ctk.CTkLabel(root, text="Введите игру:", font=("Comic Sans MS", 14))
input_label.grid(row=0, column=0, padx=10, pady=15, sticky="w")

input_entry = ctk.CTkEntry(root, placeholder_text="Введите название игры", width=250)
input_entry.grid(row=0, column=1, padx=10, pady=15, sticky="w")

process_button = ctk.CTkButton(root, text="Найти!", command=process_input, width=100)
process_button.grid(row=0, column=2, padx=10, pady=15, sticky="w")

processed_label = ctk.CTkLabel(root, text="Рекомендованные игры:", font=("Comic Sans MS", 14))
processed_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="w")

results_label = ctk.CTkLabel(root, text="Топ 10 игр:", font=("Comic Sans MS", 14))
results_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")

results_text = ctk.CTkTextbox(root, width=500, height=190, state="disabled", font=("Comic Sans MS", 12))
results_text.grid(row=3, column=0, columnspan=3, padx=10, pady=15)

# Запуск главного цикла приложения
root.mainloop()