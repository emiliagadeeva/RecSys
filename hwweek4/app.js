class MovieLensApp {
    constructor() {
        this.data = {
            ratings: null,
            movies: null,
            users: null,
            genres: null
        };
        this.models = {};
        this.isDataLoaded = false;
        this.isTraining = false;
        this.trainingHistories = {};
        
        // Genre mapping from MovieLens
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
        document.getElementById('testModels').addEventListener('click', () => this.testModels());
        document.getElementById('compareModels').addEventListener('click', () => this.compareModels());
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
            document.getElementById('testModels').disabled = false;
            document.getElementById('compareModels').disabled = false;
            
            this.updateStatus(`✅ Data loaded successfully: ${this.data.ratings.length} ratings, ${this.data.movies.size} movies, ${this.data.users ? this.data.users.size : 0} users`);
            
            setTimeout(() => {
                this.hideProgressBar();
            }, 1000);
            
        } catch (error) {
            this.updateStatus('❌ Error loading data: ' + error.message);
            console.error('Data loading error:', error);
            this.hideProgressBar();
        }
    }

    async loadMovies() {
        try {
            const response = await fetch('./data/u.item');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const text = await response.text();
            const movies = new Map();
            
            const lines = text.split('\n').filter(line => line.trim());
            let loadedCount = 0;
            
            for (const line of lines) {
                const parts = line.split('|');
                if (parts.length >= 24) {
                    const movieId = parseInt(parts[0]);
                    let title = parts[1];
                    
                    // Clean up title (remove year in parentheses if present)
                    title = title.replace(/\s*\(\d{4}\)$/, '');
                    
                    const genreVector = parts.slice(5, 24).map(x => parseInt(x));
                    const genreNames = this.getGenreNames(genreVector);
                    
                    movies.set(movieId, {
                        id: movieId,
                        title: title,
                        genres: genreVector,
                        genreNames: genreNames,
                        releaseDate: parts[2]
                    });
                    loadedCount++;
                }
            }
            
            this.data.movies = movies;
            console.log(`Loaded ${loadedCount} movies from u.item`);
            
        } catch (error) {
            console.warn('Failed to load real movies, using mock data:', error.message);
            await this.loadMockMovies();
        }
    }

    async loadRatings() {
        try {
            const response = await fetch('./data/u.data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const text = await response.text();
            const ratings = [];
            
            const lines = text.split('\n').filter(line => line.trim());
            let loadedCount = 0;
            
            for (const line of lines) {
                const parts = line.split('\t');
                if (parts.length >= 4) {
                    ratings.push({
                        userId: parseInt(parts[0]),
                        movieId: parseInt(parts[1]),
                        rating: parseInt(parts[2]),
                        timestamp: parseInt(parts[3])
                    });
                    loadedCount++;
                }
            }
            
            this.data.ratings = ratings;
            console.log(`Loaded ${loadedCount} ratings from u.data`);
            
        } catch (error) {
            console.warn('Failed to load real ratings, using mock data:', error.message);
            await this.loadMockRatings();
        }
    }

    async loadUsers() {
        try {
            const response = await fetch('./data/u.user');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const text = await response.text();
            const users = new Map();
            
            const lines = text.split('\n').filter(line => line.trim());
            let loadedCount = 0;
            
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
                    loadedCount++;
                }
            }
            
            this.data.users = users;
            console.log(`Loaded ${loadedCount} users from u.user`);
            
        } catch (error) {
            console.warn('Failed to load real users, using mock data:', error.message);
            await this.loadMockUsers();
        }
    }

    async loadMockMovies() {
        const movies = new Map();
        
        // Create realistic mock movie data
        const mockTitles = [
            "The Matrix", "Inception", "Interstellar", "The Godfather", "Pulp Fiction",
            "Forrest Gump", "Fight Club", "The Shawshank Redemption", "The Dark Knight",
            "Star Wars", "Avatar", "Titanic", "Jurassic Park", "The Avengers", "Black Panther"
        ];
        
        for (let i = 1; i <= 1682; i++) {
            const title = i <= mockTitles.length ? mockTitles[i-1] : `Movie ${i}`;
            const genreVector = Array.from({length: 19}, () => Math.random() > 0.8 ? 1 : 0);
            
            // Ensure at least one genre
            if (genreVector.every(v => v === 0)) {
                genreVector[Math.floor(Math.random() * 19)] = 1;
            }
            
            movies.set(i, {
                id: i,
                title: title,
                genres: genreVector,
                genreNames: this.getGenreNames(genreVector),
                releaseDate: '01-Jan-1995'
            });
        }
        
        this.data.movies = movies;
    }

    async loadMockRatings() {
        const ratings = [];
        
        // Create more realistic mock ratings (some patterns)
        for (let i = 0; i < 100000; i++) {
            const userId = Math.floor(Math.random() * 943) + 1;
            const movieId = Math.floor(Math.random() * 1682) + 1;
            
            // Simulate some rating patterns
            let rating;
            if (Math.random() > 0.7) {
                rating = Math.floor(Math.random() * 2) + 4; // 4-5 (high ratings)
            } else {
                rating = Math.floor(Math.random() * 3) + 2; // 2-4 (medium ratings)
            }
            
            ratings.push({
                userId: userId,
                movieId: movieId,
                rating: rating,
                timestamp: Date.now() - Math.floor(Math.random() * 1000000000)
            });
        }
        
        this.data.ratings = ratings;
    }

    async loadMockUsers() {
        const users = new Map();
        const occupations = ['educator', 'engineer', 'student', 'scientist', 'artist', 'doctor', 'lawyer'];
        const genders = ['M', 'F'];
        
        for (let i = 1; i <= 943; i++) {
            users.set(i, {
                id: i,
                age: Math.floor(Math.random() * 50) + 15,
                gender: genders[Math.floor(Math.random() * genders.length)],
                occupation: occupations[Math.floor(Math.random() * occupations.length)],
                zipCode: '00000'
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
        if (message) {
            this.updateStatus(message);
        }
        // Simulate some processing time
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    async trainAllModels() {
        if (this.isTraining) return;
        if (!this.isDataLoaded) {
            this.updateStatus('❌ Please load data first!');
            return;
        }
        
        this.isTraining = true;
        this.updateStatus('⚡ Training all models... This may take a few minutes.');
        
        try {
            // Initialize models
            this.models.baseline = new BaselineTwoTower(32, 943, 1682, 19);
            this.models.deep = new DeepLearningTwoTower(32, 943, 1682, 19);
            this.models.mlp = new MLPTwoTower(32, 943, 1682, 19);
            
            // Prepare training data
            const { userInput, movieInput, ratings } = this.prepareTrainingData();
            
            // Train models with fewer epochs for demo
            this.updateStatus('📈 Training Baseline Matrix Factorization...');
            this.trainingHistories.baseline = await this.models.baseline.train(
                userInput, movieInput, ratings, 5, 256
            );
            
            this.updateStatus('🧠 Training Deep Learning Two-Tower...');
            this.trainingHistories.deep = await this.models.deep.train(
                userInput, movieInput, ratings, 5, 256
            );
            
            this.updateStatus('🔬 Training MLP Two-Tower...');
            this.trainingHistories.mlp = await this.models.mlp.train(
                userInput, movieInput, ratings, 5, 256
            );
            
            this.plotTrainingHistories();
            this.updateStatus('✅ All models trained successfully! Ready for testing.');
            
        } catch (error) {
            this.updateStatus('❌ Error training models: ' + error.message);
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
        }
    }

    prepareTrainingData() {
        // Use only a subset for faster training in browser
        const subsetSize = Math.min(20000, this.data.ratings.length);
        const subsetRatings = this.data.ratings.slice(0, subsetSize);
        
        const userInput = subsetRatings.map(r => r.userId - 1); // 0-based indices
        const movieInput = subsetRatings.map(r => {
            const movie = this.data.movies.get(r.movieId);
            return {
                movieId: r.movieId - 1,
                genres: movie.genres
            };
        });
        const ratings = subsetRatings.map(r => r.rating >= 4 ? 1 : 0); // Binary labels
        
        return {
            userInput: tf.tensor1d(userInput, 'int32'),
            movieInput: movieInput,
            ratings: tf.tensor1d(ratings, 'float32')
        };
    }

    plotTrainingHistories() {
        const histories = [
            { name: 'Baseline MF', history: this.trainingHistories.baseline },
            { name: 'Deep Two-Tower', history: this.trainingHistories.deep },
            { name: 'MLP Two-Tower', history: this.trainingHistories.mlp }
        ];

        const lossSeries = histories.map(model => ({
            values: model.history.loss,
            name: model.name
        }));

        const accuracySeries = histories.map(model => ({
            values: model.history.acc,
            name: model.name
        }));

        // Clear previous charts
        const chartsContainer = document.getElementById('trainingCharts');
        chartsContainer.innerHTML = '<h3>📊 Training Progress</h3>';

        tfvis.show.history(
            { name: 'Training Loss', tab: 'Training', container: chartsContainer },
            lossSeries,
            ['line', 'line', 'line'],
            {
                xLabel: 'Epoch',
                yLabel: 'Loss',
                width: 400,
                height: 300
            }
        );

        tfvis.show.history(
            { name: 'Training Accuracy', tab: 'Training', container: chartsContainer },
            accuracySeries,
            ['line', 'line', 'line'],
            {
                xLabel: 'Epoch',
                yLabel: 'Accuracy',
                width: 400,
                height: 300
            }
        );
    }

    async testModels() {
        if (!this.models.mlp || !this.models.deep) {
            this.updateStatus('❌ Please train models first');
            return;
        }

        this.updateStatus('🧪 Testing models with sample user...');
        
        // Test with user ID 1 (exists in MovieLens)
        const testUserId = 1;
        const userHistory = this.getUserHistory(testUserId);
        
        if (userHistory.length === 0) {
            // If user 1 has no history, find a user with ratings
            const userWithRatings = this.findUserWithRatings();
            if (!userWithRatings) {
                this.updateStatus('❌ No users with sufficient rating history found');
                return;
            }
            testUserId = userWithRatings;
        }
        
        const mlpRecommendations = await this.getRecommendations(this.models.mlp, testUserId, 10);
        const dlRecommendations = await this.getRecommendations(this.models.deep, testUserId, 10);
        
        this.displayRecommendations(userHistory, mlpRecommendations, dlRecommendations);
        this.updateStatus('✅ Testing completed. Check comparison section below.');
    }

    getUserHistory(userId) {
        const userRatings = this.data.ratings
            .filter(r => r.userId === userId)
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 10);
        
        return userRatings.map(r => {
            const movie = this.data.movies.get(r.movieId);
            return {
                title: movie.title,
                genres: movie.genreNames,
                rating: r.rating
            };
        });
    }

    findUserWithRatings() {
        // Find a user with at least 5 ratings of 4+ stars
        const userRatingCounts = new Map();
        
        this.data.ratings.forEach(rating => {
            if (rating.rating >= 4) {
                userRatingCounts.set(rating.userId, (userRatingCounts.get(rating.userId) || 0) + 1);
            }
        });
        
        for (const [userId, count] of userRatingCounts) {
            if (count >= 5) {
                return userId;
            }
        }
        return null;
    }

    async getRecommendations(model, userId, count = 10) {
        this.updateStatus(`🔍 Generating recommendations for user ${userId}...`);
        
        const userEmbedding = await model.getUserEmbedding(userId - 1);
        const allMovies = Array.from(this.data.movies.values());
        
        const scores = [];
        const batchSize = 100;
        
        // Process in batches to avoid memory issues
        for (let i = 0; i < allMovies.length; i += batchSize) {
            const batch = allMovies.slice(i, i + batchSize);
            
            for (const movie of batch) {
                try {
                    const movieEmbedding = await model.getItemEmbedding(movie.id - 1, movie.genres);
                    const score = await this.dotProduct(userEmbedding, movieEmbedding);
                    scores.push({ movie, score });
                    movieEmbedding.dispose();
                } catch (error) {
                    console.warn(`Error processing movie ${movie.id}:`, error);
                }
            }
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
    }

    async dotProduct(vec1, vec2) {
        const product = vec1.mul(vec2);
        const sum = product.sum();
        const result = await sum.data();
        product.dispose();
        sum.dispose();
        return result[0];
    }

    displayRecommendations(history, mlpRecs, dlRecs) {
        this.populateTable('historyTable', history, true);
        this.populateTable('mlpRecommendations', mlpRecs, false);
        this.populateTable('dlRecommendations', dlRecs, false);
        
        document.getElementById('comparisonSection').style.display = 'block';
    }

    populateTable(tableId, data, isHistory) {
        const tbody = document.querySelector(`#${tableId} tbody`);
        tbody.innerHTML = '';
        
        if (data.length === 0) {
            const row = tbody.insertRow();
            const cell = row.insertCell(0);
            cell.colSpan = 3;
            cell.textContent = 'No data available';
            cell.style.textAlign = 'center';
            cell.style.color = '#666';
            return;
        }
        
        data.forEach(item => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = item.title;
            
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
            
            if (isHistory) {
                row.insertCell(2).textContent = '⭐'.repeat(item.rating);
            } else {
                row.insertCell(2).textContent = item.score.toFixed(4);
            }
        });
    }

    async compareModels() {
        if (!this.models.baseline || !this.models.mlp || !this.models.deep) {
            this.updateStatus('❌ Please train all models first');
            return;
        }

        this.updateStatus('📊 Comparing model performance...');
        
        // Calculate metrics for each model
        const baselineMetrics = await this.calculateMetrics(this.models.baseline);
        const mlpMetrics = await this.calculateMetrics(this.models.mlp);
        const deepMetrics = await this.calculateMetrics(this.models.deep);
        
        this.displayMetrics(baselineMetrics, mlpMetrics, deepMetrics);
        this.updateStatus('✅ Model comparison completed!');
    }

    async calculateMetrics(model, k = 10) {
        // Use a small sample of users for performance
        const sampleUsers = Array.from({length: 20}, (_, i) => i + 1);
        let totalPrecision = 0;
        let totalRecall = 0;
        let userCount = 0;
        
        for (const userId of sampleUsers) {
            try {
                const userRatings = this.data.ratings.filter(r => r.userId === userId && r.rating >= 4);
                if (userRatings.length < 3) continue; // Skip users with few ratings
                
                const recommendations = await this.getRecommendations(model, userId, k);
                const relevantItems = new Set(userRatings.map(r => r.movieId));
                
                const hits = recommendations.filter(rec => {
                    const movie = Array.from(this.data.movies.values()).find(m => m.title === rec.title);
                    return movie && relevantItems.has(movie.id);
                }).length;
                
                const precision = hits / k;
                const recall = hits / Math.min(userRatings.length, k);
                
                totalPrecision += precision;
                totalRecall += recall;
                userCount++;
                
            } catch (error) {
                console.warn(`Error calculating metrics for user ${userId}:`, error);
            }
        }
        
        if (userCount === 0) {
            return { precision: 0, recall: 0, f1: 0 };
        }
        
        const precision = totalPrecision / userCount;
        const recall = totalRecall / userCount;
        const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        
        return { precision, recall, f1 };
    }

    displayMetrics(baseline, mlp, deep) {
        const metricsPanel = document.getElementById('metricsPanel');
        metricsPanel.innerHTML = `
            <div class="metric-card">
                <h4>📈 Baseline MF</h4>
                <div class="metric-value">${baseline.precision.toFixed(3)}</div>
                <div><strong>Precision@10</strong></div>
                <div>Recall@10: ${baseline.recall.toFixed(3)}</div>
                <div>F1 Score: ${baseline.f1.toFixed(3)}</div>
            </div>
            <div class="metric-card">
                <h4>🔬 MLP Two-Tower</h4>
                <div class="metric-value">${mlp.precision.toFixed(3)}</div>
                <div><strong>Precision@10</strong></div>
                <div>Recall@10: ${mlp.recall.toFixed(3)}</div>
                <div>F1 Score: ${mlp.f1.toFixed(3)}</div>
            </div>
            <div class="metric-card">
                <h4>🧠 Deep Two-Tower</h4>
                <div class="metric-value">${deep.precision.toFixed(3)}</div>
                <div><strong>Precision@10</strong></div>
                <div>Recall@10: ${deep.recall.toFixed(3)}</div>
                <div>F1 Score: ${deep.f1.toFixed(3)}</div>
            </div>
        `;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log('Status:', message);
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
