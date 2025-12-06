from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import Counter

app = Flask(__name__)

# Загрузка данных
print("Загрузка данных...")
wine_df = pd.read_csv('data/wines.csv')
wine_embeddings = np.load('models/wine_embeddings.npy')

print("Загрузка моделей...")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

print("Система готова к работе!")

# Кэш для вин
wine_cache = {}

def get_wine_by_id(wine_id):
    """Получить вино по ID из кэша или датафрейма"""
    if wine_id in wine_cache:
        return wine_cache[wine_id]
    
    wine = wine_df[wine_df['id'] == wine_id].iloc[0].to_dict()
    wine_cache[wine_id] = wine
    return wine

def generate_llm_response(prompt, max_length=300):
    """Генерация ответа от LLM"""
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=llm_tokenizer.eos_token_id
        )
    
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Убираем промпт из ответа
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/wines/list')
def get_wines_list():
    """Получить список всех вин для выпадающего списка"""
    wines = wine_df.head(200).to_dict('records')  # Ограничиваем для производительности
    
    # Форматируем для фронтенда
    formatted_wines = []
    for wine in wines:
        formatted_wines.append({
            'id': wine.get('id', hash(str(wine))),
            'name': wine.get('title', 'Unknown Wine'),
            'variety': wine.get('variety', 'Unknown'),
            'country': wine.get('country', 'Unknown'),
            'price': wine.get('price', 0),
            'rating': wine.get('points', 0),
            'description': wine.get('description', 'No description available')
        })
    
    return jsonify(formatted_wines)

@app.route('/api/recommend/filtered', methods=['POST'])
def recommend_filtered():
    """Рекомендации с фильтрами"""
    try:
        data = request.json
        query = data.get('query', '')
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Фильтрация данных
        filtered_df = wine_df.copy()
        
        # Применяем фильтры
        if filters.get('variety'):
            filtered_df = filtered_df[filtered_df['variety'] == filters['variety']]
        
        if filters.get('country'):
            filtered_df = filtered_df[filtered_df['country'] == filters['country']]
        
        if filters.get('max_price'):
            filtered_df = filtered_df[filtered_df['price'] <= filters['max_price']]
        
        if len(filtered_df) == 0:
            return jsonify({'error': 'No wines match the filters', 'recommendations': []}), 200
        
        # Получаем эмбеддинг запроса
        query_embedding = embedding_model.encode(query)
        
        # Находим эмбеддинги отфильтрованных вин
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = wine_embeddings[filtered_indices]
        
        # Вычисляем схожесть
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        # Сортируем по схожести
        top_indices = np.argsort(similarities)[-20:][::-1]
        
        recommendations = []
        for idx in top_indices:
            wine = filtered_df.iloc[idx]
            recommendations.append({
                'id': wine.get('id', idx),
                'name': wine.get('title', 'Unknown Wine'),
                'variety': wine.get('variety', 'Unknown'),
                'country': wine.get('country', 'Unknown'),
                'region': wine.get('province', ''),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': wine.get('description', 'No description available'),
                'similarity_score': float(similarities[idx])
            })
        
        # Генерация LLM комментария
        prompt = f"""Пользователь ищет вино по запросу: "{query}".
Фильтры: {json.dumps(filters, ensure_ascii=False)}.
Топ-5 рекомендаций: {[r['name'] for r in recommendations[:5]]}.
Напиши краткий, дружелюбный комментарий на русском языке, объясняя почему эти вина подходят под запрос и фильтры.
Обрати внимание на особенности каждого вина."""
        
        llm_comment = generate_llm_response(prompt)
        
        return jsonify({
            'recommendations': recommendations[:15],  # Ограничиваем количество
            'llm_comment': llm_comment
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend/taste', methods=['POST'])
def recommend_by_taste():
    """Рекомендации на основе выбранных вин"""
    try:
        data = request.json
        selected_wine_ids = data.get('selected_wines', [])
        
        if not selected_wine_ids:
            return jsonify({'error': 'No wines selected'}), 400
        
        # Находим индексы выбранных вин
        selected_indices = []
        for wine_id in selected_wine_ids:
            if isinstance(wine_id, int):
                idx = wine_df[wine_df['id'] == wine_id].index
                if len(idx) > 0:
                    selected_indices.append(idx[0])
            else:
                # Если ID не найден, используем первые несколько вин
                if len(selected_indices) < 3:
                    selected_indices.append(len(selected_indices))
        
        if not selected_indices:
            return jsonify({'error': 'Selected wines not found'}), 404
        
        # Получаем эмбеддинги выбранных вин
        selected_embeddings = wine_embeddings[selected_indices]
        
        # Средний эмбеддинг предпочтений пользователя
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
            recommendations.append({
                'id': wine.get('id', idx),
                'name': wine.get('title', 'Unknown Wine'),
                'variety': wine.get('variety', 'Unknown'),
                'country': wine.get('country', 'Unknown'),
                'region': wine.get('province', ''),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': wine.get('description', 'No description available'),
                'similarity_score': float(similarities[idx])
            })
        
        # Анализ предпочтений пользователя
        selected_wines = [wine_df.iloc[idx] for idx in selected_indices]
        
        # Любимые сорта
        varieties = [w.get('variety', 'Unknown') for w in selected_wines]
        variety_counts = Counter(varieties)
        favorite_varieties = [{'variety': v, 'count': c} for v, c in variety_counts.most_common(5)]
        
        # Предпочитаемые страны
        countries = [w.get('country', 'Unknown') for w in selected_wines]
        country_counts = Counter(countries)
        preferred_countries = [{'country': c, 'count': cnt} for c, cnt in country_counts.most_common(5)]
        
        # Ценовой анализ
        prices = [float(w.get('price', 0)) for w in selected_wines]
        average_price = np.mean(prices) if prices else 0
        price_range = {'min': min(prices), 'max': max(prices)} if prices else {'min': 0, 'max': 0}
        
        # Рейтинг
        ratings = [int(w.get('points', 0)) for w in selected_wines]
        average_rating = np.mean(ratings) if ratings else 0
        
        preference_analysis = {
            'favorite_varieties': favorite_varieties,
            'preferred_countries': preferred_countries,
            'average_price': average_price,
            'price_range': price_range,
            'average_rating': average_rating
        }
        
        # Генерация LLM комментария
        selected_names = [w.get('title', 'Unknown') for w in selected_wines[:3]]
        top_recommendations = [r['name'] for r in recommendations[:3]]
        
        prompt = f"""Пользователь выбрал эти вина как любимые: {', '.join(selected_names)}.
На основе его предпочтений система рекомендует: {', '.join(top_recommendations)}.
Анализ предпочтений: пользователю нравятся сорта {favorite_varieties[0]['variety'] if favorite_varieties else 'разные'},
страны {preferred_countries[0]['country'] if preferred_countries else 'разные'},
средняя цена ${average_price:.2f}.
Напиши персональную рекомендацию на русском языке, объясни почему эти вина могут понравиться пользователю,
опиши общие характеристики и что их объединяет с выбранными винами."""
        
        llm_comment = generate_llm_response(prompt)
        
        return jsonify({
            'recommendations': recommendations[:15],
            'llm_comment': llm_comment,
            'preference_analysis': preference_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend/simple', methods=['POST'])
def recommend_simple():
    """Простые рекомендации без фильтров"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Получаем эмбеддинг запроса
        query_embedding = embedding_model.encode(query)
        
        # Вычисляем схожесть
        similarities = cosine_similarity([query_embedding], wine_embeddings)[0]
        
        # Находим топ рекомендаций
        top_indices = np.argsort(similarities)[-20:][::-1]
        
        recommendations = []
        for idx in top_indices:
            wine = wine_df.iloc[idx]
            recommendations.append({
                'id': wine.get('id', idx),
                'name': wine.get('title', 'Unknown Wine'),
                'variety': wine.get('variety', 'Unknown'),
                'country': wine.get('country', 'Unknown'),
                'region': wine.get('province', ''),
                'price': float(wine.get('price', 0)),
                'rating': int(wine.get('points', 0)),
                'description': wine.get('description', 'No description available'),
                'similarity_score': float(similarities[idx])
            })
        
        # Генерация LLM комментария
        prompt = f"""Пользователь ищет вино по запросу: "{query}".
Система рекомендует эти вина: {[r['name'] for r in recommendations[:3]]}.
Напиши краткий, дружелюбный комментарий на русском языке, почему эти вина подходят под запрос.
Опиши их особенности и в каких ситуациях они будут особенно хороши."""
        
        llm_comment = generate_llm_response(prompt)
        
        return jsonify({
            'recommendations': recommendations[:10],
            'llm_comment': llm_comment
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wine/<int:wine_id>/pairing')
def get_wine_pairing(wine_id):
    """Получить рекомендации по сочетанию с едой"""
    try:
        wine = get_wine_by_id(wine_id)
        
        prompt = f"""Вино: {wine.get('title', 'Unknown')}
Сорт: {wine.get('variety', 'Unknown')}
Страна: {wine.get('country', 'Unknown')}
Описание: {wine.get('description', '')}
Рекомендуй 3-4 блюда, которые хорошо сочетаются с этим вином.
Формат: краткий список на русском языке."""
        
        pairing = generate_llm_response(prompt, max_length=150)
        
        return jsonify({'pairing': pairing})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
