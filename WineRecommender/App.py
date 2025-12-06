from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import Counter
import gdown
import os
import requests
from io import BytesIO
import zipfile
import pickle

app = Flask(__name__)

# ==================== КОНФИГУРАЦИЯ ====================
# ЗАМЕНИТЕ ЭТИ ID НА СВОИ ФАЙЛЫ С GOOGLE DRIVE
GOOGLE_DRIVE_CONFIG = {
    'wine_csv_id': '18mwRZRlY3f6M6nN6VmiHKzDAAZxfEF7A,
    'embeddings_id': '1w7to6R0qf2h0-yBXwJl62-pRWN5LP60I',
}

# Или используйте прямые ссылки на скачивание (если файлы публичные)
DIRECT_DOWNLOAD_LINKS = {
    'wine_csv': None,  # Например: 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID'
    'embeddings': None,  # Например: 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID'
}

# Настройки кэширования
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== УТИЛИТЫ ДЛЯ ЗАГРУЗКИ ====================
def download_file_from_drive(file_id, destination, is_large_file=False):
    """
    Скачать файл с Google Drive
    """
    try:
        print(f"Скачивание файла с ID: {file_id}")
        
        # Способ 1: Используем gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        
        if is_large_file:
            gdown.download(url, destination, quiet=False, fuzzy=True, resume=True)
        else:
            gdown.download(url, destination, quiet=False, fuzzy=True)
        
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f"Файл успешно скачан: {destination}")
            return True
        else:
            print("Файл пустой или не скачан")
            return False
            
    except Exception as e:
        print(f"Ошибка при скачивании через gdown: {str(e)}")
        
        # Способ 2: Используем requests для резервного скачивания
        try:
            print("Пробуем альтернативный способ скачивания...")
            url = f'https://docs.google.com/uc?export=download&id={file_id}'
            
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            
            # Для больших файлов нужен confirmation token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    confirm_token = value
                    url = f'https://docs.google.com/uc?export=download&confirm={confirm_token}&id={file_id}'
                    response = session.get(url, stream=True, timeout=30)
                    break
            
            # Скачиваем файл
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"Прогресс: {progress:.1f}%", end='\r')
            
            print(f"\nФайл скачан: {destination}")
            return True
            
        except Exception as e2:
            print(f"Ошибка при альтернативном скачивании: {str(e2)}")
            return False

def download_file_from_url(url, destination):
    """
    Скачать файл по прямой ссылке
    """
    try:
        print(f"Скачивание файла по URL: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Прогресс: {progress:.1f}%", end='\r')
        
        print(f"\nФайл скачан: {destination}")
        return True
        
    except Exception as e:
        print(f"Ошибка при скачивании по URL: {str(e)}")
        return False

def load_wine_data():
    """
    Загрузка данных о винах из различных источников
    """
    csv_cache_path = os.path.join(CACHE_DIR, 'wines.csv')
    
    # Если файл уже есть в кэше, используем его
    if os.path.exists(csv_cache_path) and os.path.getsize(csv_cache_path) > 1000:  # Минимальный размер
        print("Загрузка данных о винах из кэша...")
        try:
            wine_df = pd.read_csv(csv_cache_path)
            print(f"Загружено {len(wine_df)} записей из кэша")
            return wine_df
        except Exception as e:
            print(f"Ошибка загрузки из кэша: {str(e)}")
    
    print("Скачивание данных о винах...")
    
    # Вариант 1: Прямая ссылка
    if DIRECT_DOWNLOAD_LINKS['wine_csv']:
        if download_file_from_url(DIRECT_DOWNLOAD_LINKS['wine_csv'], csv_cache_path):
            wine_df = pd.read_csv(csv_cache_path)
            print(f"Загружено {len(wine_df)} записей по прямой ссылке")
            return wine_df
    
    # Вариант 2: Google Drive по ID
    if GOOGLE_DRIVE_CONFIG['wine_csv_id'] and GOOGLE_DRIVE_CONFIG['wine_csv_id'] != 'YOUR_GOOGLE_DRIVE_FILE_ID_FOR_WINES_CSV':
        if download_file_from_drive(GOOGLE_DRIVE_CONFIG['wine_csv_id'], csv_cache_path, is_large_file=True):
            wine_df = pd.read_csv(csv_cache_path)
            print(f"Загружено {len(wine_df)} записей с Google Drive")
            return wine_df
    
    # Вариант 3: Создаем тестовые данные
    print("Создание тестовых данных...")
    wine_df = pd.DataFrame({
        'id': range(1, 101),
        'title': [f'Test Wine {i}' for i in range(1, 101)],
        'variety': ['Merlot', 'Cabernet Sauvignon', 'Chardonnay', 'Pinot Noir', 'Syrah'] * 20,
        'country': ['France', 'Italy', 'Spain', 'USA', 'Chile', 'Argentina', 'Australia', 'Germany'] * 12 + ['France'] * 4,
        'price': [10 + i * 2 for i in range(100)],
        'points': [80 + i % 20 for i in range(100)],
        'description': [f'This is a delicious {["red", "white", "rose"][i%3]} wine with notes of {["berries", "citrus", "chocolate", "vanilla"][i%4]}. Perfect for {["dinner", "special occasions", "everyday drinking"][i%3]}.' for i in range(100)],
        'province': ['Bordeaux', 'Tuscany', 'Rioja', 'California', 'Mendoza', 'Barossa Valley', 'Mosel'] * 14 + ['Bordeaux'] * 2,
        'winery': [f'Winery {chr(65 + i % 26)}' for i in range(100)]
    })
    
    # Сохраняем тестовые данные в кэш
    wine_df.to_csv(csv_cache_path, index=False)
    print(f"Создано {len(wine_df)} тестовых записей")
    
    return wine_df

def load_embeddings():
    """
    Загрузка эмбеддингов из различных источников
    """
    embeddings_cache_path = os.path.join(CACHE_DIR, 'wine_embeddings.npy')
    
    # Если файл уже есть в кэше, используем его
    if os.path.exists(embeddings_cache_path) and os.path.getsize(embeddings_cache_path) > 1000:
        print("Загрузка эмбеддингов из кэша...")
        try:
            embeddings = np.load(embeddings_cache_path)
            print(f"Загружено {len(embeddings)} эмбеддингов из кэша")
            return embeddings
        except Exception as e:
            print(f"Ошибка загрузки эмбеддингов из кэша: {str(e)}")
    
    print("Скачивание эмбеддингов...")
    
    # Вариант 1: Прямая ссылка на .npy файл
    if DIRECT_DOWNLOAD_LINKS['embeddings']:
        if download_file_from_url(DIRECT_DOWNLOAD_LINKS['embeddings'], embeddings_cache_path):
            embeddings = np.load(embeddings_cache_path)
            print(f"Загружено {len(embeddings)} эмбеддингов по прямой ссылке")
            return embeddings
    
    # Вариант 2: Google Drive по ID
    if GOOGLE_DRIVE_CONFIG['embeddings_id'] and GOOGLE_DRIVE_CONFIG['embeddings_id'] != 'YOUR_GOOGLE_DRIVE_FILE_ID_FOR_EMBEDDINGS':
        if download_file_from_drive(GOOGLE_DRIVE_CONFIG['embeddings_id'], embeddings_cache_path, is_large_file=True):
            embeddings = np.load(embeddings_cache_path)
            print(f"Загружено {len(embeddings)} эмбеддингов с Google Drive")
            return embeddings
    
    # Вариант 3: Создаем случайные эмбеддинги для теста
    print("Создание тестовых эмбеддингов...")
    np.random.seed(42)
    embeddings = np.random.randn(100, 768).astype(np.float32)
    
    # Сохраняем в кэш
    np.save(embeddings_cache_path, embeddings)
    print(f"Создано {len(embeddings)} тестовых эмбеддингов")
    
    return embeddings

def load_models():
    """
    Загрузка ML моделей
    """
    print("Загрузка embedding модели...")
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Embedding модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки embedding модели: {str(e)}")
        print("Используем упрощенную модель...")
        embedding_model = None
    
    print("Загрузка LLM модели...")
    try:
        # Используем более легковесную модель для демонстрации
        llm_model_name = "microsoft/DialoGPT-small"
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # Настраиваем для CPU если нет GPU
        if not torch.cuda.is_available():
            llm_model = llm_model.to('cpu')
        
        print("LLM модель загружена")
        return embedding_model, llm_tokenizer, llm_model
        
    except Exception as e:
        print(f"Ошибка загрузки LLM модели: {str(e)}")
        print("Продолжаем без LLM модели...")
        return embedding_model, None, None

# ==================== ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ ====================
print("=" * 50)
print("ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ РЕКОМЕНДАЦИЙ ВИН")
print("=" * 50)

# Загружаем данные
wine_df = load_wine_data()
wine_embeddings = load_embeddings()

# Проверяем совпадение размеров
if len(wine_df) != len(wine_embeddings):
    print(f"ВНИМАНИЕ: Размеры не совпадают! Вин: {len(wine_df)}, Эмбеддингов: {len(wine_embeddings)}")
    min_len = min(len(wine_df), len(wine_embeddings))
    wine_df = wine_df.iloc[:min_len].reset_index(drop=True)
    wine_embeddings = wine_embeddings[:min_len]
    print(f"Обрезано до {min_len} записей")

# Загружаем модели
embedding_model, llm_tokenizer, llm_model = load_models()

print("=" * 50)
print(f"СИСТЕМА ГОТОВА К РАБОТЕ")
print(f"В базе: {len(wine_df)} вин")
print(f"Размер эмбеддингов: {wine_embeddings.shape}")
print("=" * 50)

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
wine_cache = {}

def get_wine_by_id(wine_id):
    """Получить вино по ID из кэша или датафрейма"""
    if wine_id in wine_cache:
        return wine_cache[wine_id]
    
    wine = wine_df[wine_df['id'] == wine_id]
    if len(wine) == 0:
        return None
    
    wine_dict = wine.iloc[0].to_dict()
    wine_cache[wine_id] = wine_dict
    return wine_dict

def generate_llm_response(prompt, max_length=200):
    """Генерация ответа от LLM"""
    if llm_model is None or llm_tokenizer is None:
        return "LLM модель временно недоступна. Рекомендации основаны на семантическом поиске."
    
    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Перемещаем на GPU если доступно
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llm_model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=llm_tokenizer.eos_token_id
            )
        
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Убираем промпт из ответа если он есть
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"Ошибка генерации LLM: {str(e)}")
        return "Извините, не удалось сгенерировать ответ. Вот рекомендации на основе вашего запроса."

def get_query_embedding(query):
    """Получить эмбеддинг для текстового запроса"""
    if embedding_model is None:
        # Возвращаем случайный эмбеддинг если модель не загружена
        np.random.seed(hash(query) % 10000)
        return np.random.randn(768).astype(np.float32)
    
    return embedding_model.encode(query)

# ==================== FLASK МАРШРУТЫ ====================
@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Проверка статуса системы"""
    return jsonify({
        'status': 'ready',
        'wines_count': len(wine_df),
        'embeddings_shape': wine_embeddings.shape,
        'embedding_model_loaded': embedding_model is not None,
        'llm_model_loaded': llm_model is not None
    })

@app.route('/api/wines/list')
def get_wines_list():
    """Получить список всех вин для выпадающего списка"""
    try:
        # Ограничиваем количество для производительности
        wines = wine_df.head(500).to_dict('records')
        
        formatted_wines = []
        for wine in wines:
            formatted_wines.append({
                'id': int(wine.get('id', 0)),
                'name': str(wine.get('title', 'Unknown Wine')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': str(wine.get('description', 'No description available'))
            })
        
        return jsonify({
            'success': True,
            'wines': formatted_wines,
            'total': len(formatted_wines)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/wines/filters')
def get_filters():
    """Получить доступные фильтры"""
    try:
        varieties = wine_df['variety'].dropna().unique().tolist() if 'variety' in wine_df.columns else []
        countries = wine_df['country'].dropna().unique().tolist() if 'country' in wine_df.columns else []
        
        prices = wine_df['price'].dropna()
        price_range = {
            'min': float(prices.min()) if len(prices) > 0 else 0,
            'max': float(prices.max()) if len(prices) > 0 else 100,
            'avg': float(prices.mean()) if len(prices) > 0 else 50
        }
        
        return jsonify({
            'success': True,
            'varieties': varieties[:50],  # Ограничиваем количество
            'countries': countries[:50],
            'price_range': price_range
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend/simple', methods=['POST'])
def recommend_simple():
    """Простые рекомендации по текстовому запросу"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Пожалуйста, введите описание вина, которое вы ищете'
            }), 400
        
        print(f"Поиск рекомендаций для запроса: '{query}'")
        
        # Получаем эмбеддинг запроса
        query_embedding = get_query_embedding(query)
        
        # Вычисляем косинусную схожесть
        similarities = cosine_similarity([query_embedding], wine_embeddings)[0]
        
        # Получаем топ-20 наиболее похожих вин
        top_indices = np.argsort(similarities)[-30:][::-1]
        
        recommendations = []
        for idx in top_indices[:20]:  # Берем топ-20
            wine = wine_df.iloc[idx]
            wine_id = wine.get('id', idx)
            
            recommendations.append({
                'id': int(wine_id),
                'name': str(wine.get('title', f'Wine {wine_id}')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'region': str(wine.get('province', '')),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': str(wine.get('description', '')),
                'similarity_score': float(similarities[idx]),
                'winery': str(wine.get('winery', '')) if 'winery' in wine else ''
            })
        
        # Сортируем по схожести (на всякий случай)
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Генерируем LLM комментарий
        llm_comment = ""
        if len(recommendations) > 0:
            top_wines = [r['name'] for r in recommendations[:3]]
            prompt = f"""Пользователь ищет вино по запросу: "{query}".
Я рекомендую эти вина: {', '.join(top_wines)}.
Напиши краткий, дружелюбный комментарий на русском языке, объясняя почему эти вина подходят под запрос.
Обрати внимание на их особенности."""
            
            llm_comment = generate_llm_response(prompt)
        
        return jsonify({
            'success': True,
            'query': query,
            'recommendations': recommendations[:15],  # Отправляем топ-15
            'llm_comment': llm_comment,
            'total_found': len(recommendations)
        })
        
    except Exception as e:
        print(f"Ошибка в recommend_simple: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/recommend/filtered', methods=['POST'])
def recommend_filtered():
    """Рекомендации с фильтрами"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        filters = data.get('filters', {})
        
        print(f"Поиск с фильтрами: '{query}', фильтры: {filters}")
        
        # Начинаем со всех вин
        filtered_df = wine_df.copy()
        
        # Применяем фильтры
        if filters.get('variety'):
            filtered_df = filtered_df[filtered_df['variety'] == filters['variety']]
        
        if filters.get('country'):
            filtered_df = filtered_df[filtered_df['country'] == filters['country']]
        
        if filters.get('max_price'):
            try:
                max_price = float(filters['max_price'])
                filtered_df = filtered_df[filtered_df['price'] <= max_price]
            except:
                pass
        
        if filters.get('min_rating'):
            try:
                min_rating = int(filters['min_rating'])
                filtered_df = filtered_df[filtered_df['points'] >= min_rating]
            except:
                pass
        
        if len(filtered_df) == 0:
            return jsonify({
                'success': True,
                'query': query,
                'recommendations': [],
                'llm_comment': 'К сожалению, нет вин, соответствующих вашим фильтрам. Попробуйте изменить параметры поиска.',
                'total_found': 0
            })
        
        # Если запрос пустой, просто возвращаем случайные вина из отфильтрованных
        if not query:
            sample_df = filtered_df.sample(min(20, len(filtered_df)))
            recommendations = []
            
            for _, wine in sample_df.iterrows():
                wine_id = wine.get('id', 0)
                recommendations.append({
                    'id': int(wine_id),
                    'name': str(wine.get('title', f'Wine {wine_id}')),
                    'variety': str(wine.get('variety', 'Unknown')),
                    'country': str(wine.get('country', 'Unknown')),
                    'region': str(wine.get('province', '')),
                    'price': float(wine.get('price', 0)),
                    'rating': int(wine.get('points', 0)),
                    'description': str(wine.get('description', '')),
                    'similarity_score': 0.5,
                    'winery': str(wine.get('winery', '')) if 'winery' in wine else ''
                })
            
            return jsonify({
                'success': True,
                'query': '',
                'recommendations': recommendations,
                'llm_comment': f'Вот {len(recommendations)} вин, соответствующих вашим фильтрам.',
                'total_found': len(recommendations)
            })
        
        # Получаем эмбеддинг запроса
        query_embedding = get_query_embedding(query)
        
        # Получаем эмбеддинги отфильтрованных вин
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = wine_embeddings[filtered_indices]
        
        # Вычисляем схожесть
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        # Сортируем по схожести
        top_indices = np.argsort(similarities)[-20:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx >= len(filtered_df):
                continue
                
            wine = filtered_df.iloc[idx]
            wine_id = wine.get('id', 0)
            
            recommendations.append({
                'id': int(wine_id),
                'name': str(wine.get('title', f'Wine {wine_id}')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'region': str(wine.get('province', '')),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': str(wine.get('description', '')),
                'similarity_score': float(similarities[idx]),
                'winery': str(wine.get('winery', '')) if 'winery' in wine else ''
            })
        
        # Генерация LLM комментария
        llm_comment = ""
        if len(recommendations) > 0:
            top_wines = [r['name'] for r in recommendations[:3]]
            filter_desc = []
            if filters.get('variety'):
                filter_desc.append(f"сорт: {filters['variety']}")
            if filters.get('country'):
                filter_desc.append(f"страна: {filters['country']}")
            if filters.get('max_price'):
                filter_desc.append(f"максимальная цена: ${filters['max_price']}")
            
            filter_str = ', '.join(filter_desc) if filter_desc else "без дополнительных фильтров"
            
            prompt = f"""Пользователь ищет вино по запросу: "{query}".
Фильтры: {filter_str}.
Топ-3 рекомендации: {', '.join(top_wines)}.
Напиши краткий, дружелюбный комментарий на русском языке, объясняя почему эти вина подходят под запрос и фильтры."""
            
            llm_comment = generate_llm_response(prompt)
        
        return jsonify({
            'success': True,
            'query': query,
            'recommendations': recommendations[:15],
            'llm_comment': llm_comment,
            'total_found': len(recommendations)
        })
        
    except Exception as e:
        print(f"Ошибка в recommend_filtered: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/recommend/taste', methods=['POST'])
def recommend_by_taste():
    """Рекомендации на основе выбранных вин (коллаборативная фильтрация)"""
    try:
        data = request.json
        selected_wine_ids = data.get('selected_wines', [])
        
        if not selected_wine_ids or len(selected_wine_ids) == 0:
            return jsonify({
                'success': False,
                'error': 'Пожалуйста, выберите хотя бы одно вино, которое вам нравится'
            }), 400
        
        print(f"Рекомендации на основе выбранных вин: {selected_wine_ids}")
        
        # Находим индексы выбранных вин
        selected_indices = []
        valid_wines = []
        
        for wine_id in selected_wine_ids:
            try:
                wine_idx = wine_df[wine_df['id'] == wine_id].index
                if len(wine_idx) > 0:
                    selected_indices.append(wine_idx[0])
                    valid_wines.append(wine_df.iloc[wine_idx[0]])
            except:
                continue
        
        if not selected_indices:
            return jsonify({
                'success': False,
                'error': 'Не удалось найти выбранные вина в базе данных'
            }), 404
        
        # Вычисляем средний эмбеддинг предпочтений пользователя
        selected_embeddings = wine_embeddings[selected_indices]
        user_embedding = np.mean(selected_embeddings, axis=0)
        
        # Вычисляем схожесть со всеми винами
        similarities = cosine_similarity([user_embedding], wine_embeddings)[0]
        
        # Исключаем уже выбранные вина
        for idx in selected_indices:
            similarities[idx] = -1
        
        # Находим топ рекомендаций
        top_indices = np.argsort(similarities)[-30:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx in selected_indices:
                continue
                
            wine = wine_df.iloc[idx]
            wine_id = wine.get('id', 0)
            
            recommendations.append({
                'id': int(wine_id),
                'name': str(wine.get('title', f'Wine {wine_id}')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'region': str(wine.get('province', '')),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': str(wine.get('description', '')),
                'similarity_score': float(similarities[idx]),
                'winery': str(wine.get('winery', '')) if 'winery' in wine else ''
            })
        
        # Анализ предпочтений пользователя
        favorite_varieties = []
        preferred_countries = []
        
        if len(valid_wines) > 0:
            # Любимые сорта
            varieties = [w.get('variety', 'Unknown') for w in valid_wines]
            from collections import Counter
            variety_counts = Counter(varieties)
            favorite_varieties = [{'variety': v, 'count': c} for v, c in variety_counts.most_common(3)]
            
            # Предпочитаемые страны
            countries = [w.get('country', 'Unknown') for w in valid_wines]
            country_counts = Counter(countries)
            preferred_countries = [{'country': c, 'count': cnt} for c, cnt in country_counts.most_common(3)]
            
            # Ценовой диапазон
            prices = [float(w.get('price', 0)) for w in valid_wines]
            price_range = {
                'min': float(min(prices)) if prices else 0,
                'max': float(max(prices)) if prices else 0,
                'avg': float(np.mean(prices)) if prices else 0
            }
            
            # Средний рейтинг
            ratings = [int(w.get('points', 0)) for w in valid_wines]
            avg_rating = float(np.mean(ratings)) if ratings else 0
        else:
            price_range = {'min': 0, 'max': 0, 'avg': 0}
            avg_rating = 0
        
        preference_analysis = {
            'favorite_varieties': favorite_varieties,
            'preferred_countries': preferred_countries,
            'price_range': price_range,
            'average_rating': avg_rating
        }
        
        # Генерация LLM комментария
        llm_comment = ""
        if len(recommendations) > 0 and len(valid_wines) > 0:
            selected_names = [w.get('title', 'Unknown') for w in valid_wines[:3]]
            top_recommendations = [r['name'] for r in recommendations[:3]]
            
            prompt = f"""На основе того, что пользователю нравятся эти вина: {', '.join(selected_names)}.
Я рекомендую: {', '.join(top_recommendations)}.
Напиши персональную рекомендацию на русском языке, объясни почему эти вина могут понравиться пользователю.
Упомяни общие характеристики и почему они могут подойти."""
            
            llm_comment = generate_llm_response(prompt)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations[:15],
            'llm_comment': llm_comment,
            'preference_analysis': preference_analysis,
            'total_found': len(recommendations)
        })
        
    except Exception as e:
        print(f"Ошибка в recommend_by_taste: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/wine/<int:wine_id>')
def get_wine_details(wine_id):
    """Получить детальную информацию о вине"""
    try:
        wine = get_wine_by_id(wine_id)
        
        if wine is None:
            return jsonify({
                'success': False,
                'error': f'Вино с ID {wine_id} не найдено'
            }), 404
        
        # Находим похожие вина
        wine_idx = wine_df[wine_df['id'] == wine_id].index
        if len(wine_idx) > 0:
            idx = wine_idx[0]
            if idx < len(wine_embeddings):
                wine_embedding = wine_embeddings[idx]
                similarities = cosine_similarity([wine_embedding], wine_embeddings)[0]
                
                # Исключаем текущее вино
                similarities[idx] = -1
                
                # Находим топ-5 похожих вин
                similar_indices = np.argsort(similarities)[-6:][::-1]
                similar_wines = []
                
                for sim_idx in similar_indices[:5]:  # Берем топ-5
                    if sim_idx != idx and sim_idx < len(wine_df):
                        sim_wine = wine_df.iloc[sim_idx]
                        similar_wines.append({
                            'id': int(sim_wine.get('id', 0)),
                            'name': str(sim_wine.get('title', '')),
                            'variety': str(sim_wine.get('variety', '')),
                            'country': str(sim_wine.get('country', '')),
                            'price': float(sim_wine.get('price', 0)),
                            'similarity_score': float(similarities[sim_idx])
                        })
            else:
                similar_wines = []
        else:
            similar_wines = []
        
        return jsonify({
            'success': True,
            'wine': wine,
            'similar_wines': similar_wines
        })
        
    except Exception as e:
        print(f"Ошибка в get_wine_details: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/wine/<int:wine_id>/pairing')
def get_wine_pairing(wine_id):
    """Получить рекомендации по сочетанию с едой"""
    try:
        wine = get_wine_by_id(wine_id)
        
        if wine is None:
            return jsonify({
                'success': False,
                'error': f'Вино с ID {wine_id} не найдено'
            }), 404
        
        # Генерируем рекомендации по сочетанию с едой
        wine_name = wine.get('title', 'это вино')
        wine_variety = wine.get('variety', '')
        wine_country = wine.get('country', '')
        wine_description = wine.get('description', '')
        
        prompt = f"""Вино: {wine_name}
Сорт: {wine_variety}
Страна: {wine_country}
Описание: {wine_description[:200]}
Рекомендуй 3-4 блюда, которые хорошо сочетаются с этим вином.
Формат: краткий список на русском языке, без лишних объяснений."""
        
        pairing = generate_llm_response(prompt, max_length=150)
        
        return jsonify({
            'success': True,
            'pairing': pairing
        })
        
    except Exception as e:
        print(f"Ошибка в get_wine_pairing: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Страница не найдена'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Внутренняя ошибка сервера'
    }), 500

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Запуск Flask сервера...")
    print(f"Ссылка: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
