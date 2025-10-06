class MovieLensApp {
    constructor() {
        this.data = {
            ratings: null,
            movies: null,
            users: null
        };
        this.models = {};
        this.isDataLoaded = false;
        this.isTraining = false;
        this.trainingHistories = {};
        
        this.genreList = [
            'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ];
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModels').addEventListener('click', () => this.trainAllModels());
        document.getElementById('compareModels').addEventListener('click', () => this.compareAllModels());
    }

    async loadData() {
        this.updateStatus('📥 Loading MovieLens 100K dataset...');
        this.showProgressBar();
        
        try {
            await this.updateProgress(10, 'Loading movies data...');
            await this.loadMovies();
            
            await this.updateProgress(40, 'Loading ratings data...');
            await this.loadRatings();
            
            await this.updateProgress(70, 'Loading user data...');
            await this.loadUsers();
            
            await this.updateProgress(100, 'Data processing completed!');
            
            this.isDataLoaded = true;
            document.getElementById('trainModels').disabled = false;
            document.getElementById('compareModels').disabled = false;
            
            const movieCount = this.data.movies.size;
            const ratingCount = this.data.ratings.length;
            const userCount = this.data.users.size;
            
            this.updateStatus(`✅ Data loaded: ${ratingCount} ratings, ${movieCount} movies, ${userCount} users`);
            
            setTimeout(() => this.hideProgressBar(), 1000);
            
        } catch (error) {
            this.updateStatus('❌ Error loading data: ' + error.message);
            console.error('Data loading error:', error);
            this.hideProgressBar();
        }
    }

    async loadMovies() {
        try {
            const response = await fetch('./data/u.item');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const text = await response.text();
            const movies = new Map();
            
            const lines = text.split('\n').filter(line => line.trim());
            
            for (const line of lines) {
                const parts = line.split('|');
                if (parts.length >= 24) {
                    const movieId = parseInt(parts[0]);
                    let title = parts[1].replace(/\s*\(\d{4}\)$/, '');
                    const genreVector = parts.slice(5, 24).map(x => parseInt(x));
                    
                    movies.set(movieId, {
                        id: movieId,
                        title: title,
                        genres: genreVector,
                        genreNames: this.getGenreNames(genreVector)
                    });
                }
            }
            
            this.data.movies = movies;
            console.log(`Loaded ${movies.size} movies`);
            
        } catch (error) {
            console.warn('Failed to load real movies, using mock data');
            await this.loadMockMovies();
        }
    }

    async loadRatings() {
        try {
            const response = await fetch('./data/u.data');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const text = await response.text();
            const ratings = [];
            
            const lines = text.split('\n').filter(line => line.trim());
            
            for (const line of lines) {
                const parts = line.split('\t');
                if (parts.length >= 4) {
                    ratings.push({
                        userId: parseInt(parts[0]),
                        movieId: parseInt(parts[1]),
                        rating: parseInt(parts[2]),
                        timestamp: parseInt(parts[3])
                    });
                }
            }
            
            this.data.ratings = ratings;
            console.log(`Loaded ${ratings.length} ratings`);
            
        } catch (error) {
            console.warn('Failed to load real ratings, using mock data');
            await this.loadMockRatings();
        }
    }

    async loadUsers() {
        try {
            const response = await fetch('./data/u.user');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const text = await response.text();
            const users = new Map();
            
            const lines = text.split('\n').filter(line => line.trim());
            
            for (const line of lines) {
                const parts = line.split('|');
                if (parts.length >= 5) {
                    users.set(parseInt(parts[0]), {
                        id: parseInt(parts[0]),
                        age: parseInt(parts[1]),
                        gender: parts[2],
                        occupation: parts[3],
                        zipCode: parts[4]
                    });
                }
            }
            
            this.data.users = users;
            console.log(`Loaded ${users.size} users`);
            
        } catch (error) {
            console.warn('Failed to load real users, using mock data');
            await this.loadMockUsers();
        }
    }

    async loadMockMovies() {
        const movies = new Map();
        const mockTitles = [
            "The Matrix", "Inception", "Interstellar", "The Godfather", "Pulp Fiction",
            "Forrest Gump", "Fight Club", "The Shawshank Redemption", "The Dark Knight"
        ];
        
        for (let i = 1; i <= 500; i++) {
            const title = i <= mockTitles.length ? mockTitles[i-1] : `Movie ${i}`;
            const genreVector = Array.from({length: 19}, () => Math.random() > 0.8 ? 1 : 0);
            if (genreVector.every(v => v === 0)) genreVector[Math.floor(Math.random() * 19)] = 1;
            
            movies.set(i, {
                id: i,
                title: title,
                genres: genreVector,
                genreNames: this.getGenreNames(genreVector)
            });
        }
        
        this.data.movies = movies;
    }

    async loadMockRatings() {
        const ratings = [];
        for (let i = 0; i < 2000; i++) {
            ratings.push({
                userId: Math.floor(Math.random() * 200) + 1,
                movieId: Math.floor(Math.random() * 500) + 1,
                rating: Math.random() > 0.3 ? Math.floor(Math.random() * 2) + 4 : Math.floor(Math.random() * 3) + 2,
                timestamp: Date.now()
            });
        }
        this.data.ratings = ratings;
    }

    async loadMockUsers() {
        const users = new Map();
        const occupations = ['educator', 'engineer', 'student', 'artist'];
        for (let i = 1; i <= 200; i++) {
            users.set(i, {
                id: i,
                age: Math.floor(Math.random() * 50) + 15,
                gender: Math.random() > 0.5 ? 'M' : 'F',
                occupation: occupations[Math.floor(Math.random() * occupations.length)]
            });
        }
        this.data.users = users;
    }

    getGenreNames(genreVector) {
        return genreVector
            .map((value, index) => value === 1 ? this.genreList[index] : null)
            .filter(name => name !== null);
    }

    showProgressBar() {
        document.getElementById('progressBar').style.display = 'block';
        document.getElementById('progressFill').style.width = '0%';
    }

    hideProgressBar() {
        document.getElementById('progressBar').style.display = 'none';
    }

    async updateProgress(percent, message) {
        document.getElementById('progressFill').style.width = percent + '%';
        if (message) this.updateStatus(message);
        await new Promise(resolve => setTimeout(resolve, 50));
    }

    async trainAllModels() {
        if (this.isTraining || !this.isDataLoaded) {
            this.updateStatus('❌ Please load data first!');
            return;
        }
        
        this.isTraining = true;
        this.updateStatus('⚡ Training all three models...');
        
        try {
            // ВАЖНО: Используем правильные имена классов из two-tower.js
            this.models = {
                baseline: new WithoutDLTwoTower(16, 200, 500, 19),
                mlp: new MLPTwoTower(16, 200, 500, 19),
                deep: new DeepLearningTwoTower(16, 200, 500, 19)
            };
            
            const { userInput, movieInput, ratings } = this.prepareTrainingData();
            
            // Train models sequentially
            this.updateStatus('📈 Training Without DL model...');
            this.trainingHistories.baseline = await this.models.baseline.train(userInput, movieInput, ratings, 2, 32);
            
            this.updateStatus('🔬 Training MLP Two-Tower...');
            this.trainingHistories.mlp = await this.models.mlp.train(userInput, movieInput, ratings, 2, 32);
            
            this.updateStatus('🧠 Training Deep Learning Two-Tower...');
            this.trainingHistories.deep = await this.models.deep.train(userInput, movieInput, ratings, 2, 32);
            
            this.plotTrainingHistories();
            this.updateStatus('✅ All models trained! Click "Compare All Models" to see results.');
            
        } catch (error) {
            this.updateStatus('❌ Error training models: ' + error.message);
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
        }
    }

    prepareTrainingData() {
        const subsetSize = Math.min(500, this.data.ratings.length);
        const subsetRatings = this.data.ratings.slice(0, subsetSize);
        
        const userInput = subsetRatings.map(r => r.userId - 1);
        const movieInput = subsetRatings.map(r => {
            const movie = this.data.movies.get(r.movieId);
            return {
                movieId: r.movieId - 1,
                genres: movie.genres
            };
        });
        const ratings = subsetRatings.map(r => r.rating >= 4 ? 1 : 0);
        
        console.log(`Training data: ${userInput.length} samples`);
        
        return {
            userInput: tf.tensor1d(userInput, 'int32'),
            movieInput: movieInput,
            ratings: tf.tensor1d(ratings, 'float32')
        };
    }

    plotTrainingHistories() {
        const histories = [
            { name: 'Without DL', history: this.trainingHistories.baseline, color: '#007cba' },
            { name: 'MLP Two-Tower', history: this.trainingHistories.mlp, color: '#28a745' },
            { name: 'Deep Two-Tower', history: this.trainingHistories.deep, color: '#dc3545' }
        ].filter(model => model.history && model.history.loss.length > 0);
        
        if (histories.length === 0) return;

        const lossSeries = histories.map(model => ({
            values: model.history.loss,
            name: model.name
        }));

        const chartsContainer = document.getElementById('trainingCharts');
        chartsContainer.innerHTML = '<h3>📊 Training Progress - Loss Curves</h3>';

        tfvis.show.history(
            { name: 'Model Training Loss', tab: 'Training' },
            lossSeries,
            ['line', 'line', 'line'].slice(0, histories.length),
            {
                xLabel: 'Epoch',
                yLabel: 'Loss',
                width: 600,
                height: 400
            },
            chartsContainer
        );
    }

    async compareAllModels() {
        if (!this.models.baseline || !this.models.mlp || !this.models.deep) {
            this.updateStatus('❌ Please train all models first');
            return;
        }

        this.updateStatus('📊 Comparing all three models...');
        
        try {
            const testUserId = this.findUserWithRatings();
            if (!testUserId) {
                this.updateStatus('❌ No suitable test user found');
                return;
            }
            
            // Get user history
            const userHistory = this.getUserHistory(testUserId);
            this.displayUserHistory(userHistory);
            
            // Get recommendations from all models
            const [baselineRecs, mlpRecs, deepRecs] = await Promise.all([
                this.getRecommendations(this.models.baseline, testUserId, 8),
                this.getRecommendations(this.models.mlp, testUserId, 8),
                this.getRecommendations(this.models.deep, testUserId, 8)
            ]);
            
            // Display all recommendations
            this.displayRecommendations(baselineRecs, mlpRecs, deepRecs);
            
            // Calculate and display metrics
            const metrics = await this.calculateAllMetrics(testUserId);
            this.displayMetrics(metrics);
            
            document.getElementById('comparisonSection').style.display = 'block';
            this.updateStatus('✅ Model comparison completed!');
            
        } catch (error) {
            this.updateStatus('❌ Error comparing models: ' + error.message);
            console.error('Comparison error:', error);
        }
    }

    getUserHistory(userId) {
        return this.data.ratings
            .filter(r => r.userId === userId)
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 10)
            .map(r => {
                const movie = this.data.movies.get(r.movieId);
                return {
                    title: movie ? movie.title : `Movie ${r.movieId}`,
                    genres: movie ? movie.genreNames : ['Unknown'],
                    rating: r.rating
                };
            });
    }

    displayUserHistory(history) {
        const tbody = document.querySelector('#historyTable tbody');
        tbody.innerHTML = '';
        
        history.forEach(item => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = item.title;
            
            const genreCell = row.insertCell(1);
            item.genres.forEach(genre => {
                const span = document.createElement('span');
                span.className = 'genre-tag';
                span.textContent = genre;
                genreCell.appendChild(span);
            });
            
            const ratingCell = row.insertCell(2);
            ratingCell.textContent = '⭐'.repeat(item.rating);
            ratingCell.title = `Rating: ${item.rating}/5`;
        });
    }

    findUserWithRatings() {
        const userRatingCounts = new Map();
        this.data.ratings.forEach(rating => {
            if (rating.rating >= 4) {
                userRatingCounts.set(rating.userId, (userRatingCounts.get(rating.userId) || 0) + 1);
            }
        });
        
        for (const [userId, count] of userRatingCounts) {
            if (count >= 2) return userId;
        }
        
        const usersWithRatings = new Set(this.data.ratings.map(r => r.userId));
        return usersWithRatings.size > 0 ? Array.from(usersWithRatings)[0] : 1;
    }

    async getRecommendations(model, userId, count = 8) {
        try {
            const userEmbedding = await model.getUserEmbedding(userId - 1);
            const allMovies = Array.from(this.data.movies.values());
            const scores = [];
            
            // Test with smaller subset for performance
            for (const movie of allMovies.slice(0, 50)) {
                const movieEmbedding = await model.getItemEmbedding(movie.id - 1, movie.genres);
                const score = await this.dotProduct(userEmbedding, movieEmbedding);
                scores.push({ movie, score });
                movieEmbedding.dispose();
            }
            
            userEmbedding.dispose();
            
            return scores
                .sort((a, b) => b.score - a.score)
                .slice(0, count)
                .map(item => ({
                    title: item.movie.title,
                    genres: item.movie.genreNames,
                    score: item.score
                }));
                
        } catch (error) {
            console.error('Error generating recommendations:', error);
            return [];
        }
    }

    async dotProduct(vec1, vec2) {
        const product = vec1.mul(vec2);
        const sum = product.sum();
        const result = await sum.data();
        tf.dispose([product, sum]);
        return result[0];
    }

    displayRecommendations(baselineRecs, mlpRecs, deepRecs) {
        this.populateRecommendationTable('baselineRecommendations', baselineRecs);
        this.populateRecommendationTable('mlpRecommendations', mlpRecs);
        this.populateRecommendationTable('deepRecommendations', deepRecs);
    }

    populateRecommendationTable(tableId, recommendations) {
        const tbody = document.getElementById(tableId);
        tbody.innerHTML = '';
        
        if (recommendations.length === 0) {
            const row = tbody.insertRow();
            const cell = row.insertCell(0);
            cell.colSpan = 3;
            cell.textContent = 'No recommendations available';
            cell.style.textAlign = 'center';
            cell.style.color = '#666';
            cell.style.padding = '20px';
            return;
        }
        
        recommendations.forEach(item => {
            const row = tbody.insertRow();
            
            // Title cell
            const titleCell = row.insertCell(0);
            titleCell.textContent = item.title;
            titleCell.style.maxWidth = '150px';
            titleCell.style.overflow = 'hidden';
            titleCell.style.textOverflow = 'ellipsis';
            titleCell.style.whiteSpace = 'nowrap';
            
            // Genre cell
            const genreCell = row.insertCell(1);
            if (item.genres && item.genres.length > 0) {
                item.genres.forEach(genre => {
                    const span = document.createElement('span');
                    span.className = 'genre-tag';
                    span.textContent = genre;
                    genreCell.appendChild(span);
                });
            } else {
                genreCell.textContent = 'Unknown';
                genreCell.style.color = '#999';
            }
            
            // Score cell
            const scoreCell = row.insertCell(2);
            scoreCell.textContent = item.score.toFixed(4);
            scoreCell.className = 'score-cell';
            scoreCell.style.fontFamily = 'monospace';
        });
    }

    async calculateAllMetrics(testUserId) {
        const k = 5;
        const userRatings = this.data.ratings.filter(r => r.userId === testUserId && r.rating >= 4);
        const relevantItems = new Set(userRatings.map(r => r.movieId));
        
        const metrics = {};
        
        for (const [modelName, model] of Object.entries(this.models)) {
            try {
                const recommendations = await this.getRecommendations(model, testUserId, k);
                
                let hits = 0;
                recommendations.forEach(rec => {
                    const movie = Array.from(this.data.movies.values()).find(m => m.title === rec.title);
                    if (movie && relevantItems.has(movie.id)) {
                        hits++;
                    }
                });
                
                const precision = hits / k;
                const recall = userRatings.length > 0 ? hits / Math.min(userRatings.length, k) : 0;
                const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
                
                metrics[modelName] = { precision, recall, f1 };
            } catch (error) {
                console.warn(`Error calculating metrics for ${modelName}:`, error);
                metrics[modelName] = { precision: 0, recall: 0, f1: 0 };
            }
        }
        
        return metrics;
    }

    displayMetrics(metrics) {
        const metricsPanel = document.getElementById('metricsPanel');
        
        const metricCards = {
            baseline: { name: '📈 Without DL', class: 'baseline' },
            mlp: { name: '🔬 MLP Two-Tower', class: 'mlp' },
            deep: { name: '🧠 Deep Two-Tower', class: 'deep' }
        };
        
        metricsPanel.innerHTML = Object.entries(metricCards).map(([key, config]) => {
            const metric = metrics[key] || { precision: 0, recall: 0, f1: 0 };
            return `
                <div class="metric-card ${config.class}">
                    <h4>${config.name}</h4>
                    <div class="metric-value">${metric.precision.toFixed(3)}</div>
                    <div><strong>Precision@5</strong></div>
                    <div>Recall@5: ${metric.recall.toFixed(3)}</div>
                    <div>F1 Score: ${metric.f1.toFixed(3)}</div>
                </div>
            `;
        }).join('');
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log('Status:', message);
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
    console.log('MovieLens Three-Model Comparison Demo initialized');
});
