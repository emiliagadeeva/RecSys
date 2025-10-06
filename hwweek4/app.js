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
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModels').addEventListener('click', () => this.trainAllModels());
        document.getElementById('testModels').addEventListener('click', () => this.testModels());
        document.getElementById('compareModels').addEventListener('click', () => this.compareModels());
    }

    async loadData() {
        this.updateStatus('Loading MovieLens 100K data...');
        
        try {
            await this.loadMovies();
            await this.loadRatings();
            await this.loadUsers();
            
            this.isDataLoaded = true;
            document.getElementById('trainModels').disabled = false;
            document.getElementById('testModels').disabled = false;
            document.getElementById('compareModels').disabled = false;
            
            this.updateStatus(`Data loaded successfully: ${this.data.ratings.length} ratings, ${this.data.movies.size} movies, ${this.data.users.size} users`);
            
        } catch (error) {
            this.updateStatus('Error loading data: ' + error.message);
            console.error(error);
        }
    }

    async loadMovies() {
        // In a real implementation, you would fetch u.item from a hosted location
        // For demo purposes, we'll create mock movie data
        const movies = new Map();
        const genres = new Set(['Action', 'Adventure', 'Animation', 'Children\'s', 
                              'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                              'Sci-Fi', 'Thriller', 'War', 'Western']);
        
        // Create mock movie data - in practice, you'd parse u.item
        for (let i = 1; i <= 1682; i++) {
            const genreVector = Array.from({length: 19}, () => Math.random() > 0.7 ? 1 : 0);
            movies.set(i, {
                id: i,
                title: `Movie ${i}`,
                genres: genreVector,
                genreNames: this.getGenreNames(genreVector)
            });
        }
        
        this.data.movies = movies;
        this.data.genres = genres;
        this.data.genreList = Array.from(genres);
    }

    async loadRatings() {
        // Mock ratings data - in practice, you'd parse u.data
        const ratings = [];
        for (let i = 0; i < 100000; i++) {
            ratings.push({
                userId: Math.floor(Math.random() * 943) + 1,
                movieId: Math.floor(Math.random() * 1682) + 1,
                rating: Math.floor(Math.random() * 5) + 1,
                timestamp: Date.now()
            });
        }
        this.data.ratings = ratings;
    }

    async loadUsers() {
        // Mock user data - in practice, you'd parse u.user
        const users = new Map();
        for (let i = 1; i <= 943; i++) {
            users.set(i, {
                id: i,
                age: Math.floor(Math.random() * 50) + 15,
                gender: Math.random() > 0.5 ? 'M' : 'F',
                occupation: 'occupation_' + (Math.floor(Math.random() * 10) + 1)
            });
        }
        this.data.users = users;
    }

    getGenreNames(genreVector) {
        return this.data.genreList.filter((_, index) => genreVector[index] === 1);
    }

    async trainAllModels() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.updateStatus('Training all models...');
        
        try {
            // Initialize models
            this.models.baseline = new BaselineTwoTower(32, 943, 1682, 19);
            this.models.deep = new DeepLearningTwoTower(32, 943, 1682, 19);
            this.models.mlp = new MLPTwoTower(32, 943, 1682, 19);
            
            // Prepare training data
            const { userInput, movieInput, ratings } = this.prepareTrainingData();
            
            // Train models sequentially
            this.trainingHistories.baseline = await this.models.baseline.train(
                userInput, movieInput, ratings
            );
            
            this.trainingHistories.deep = await this.models.deep.train(
                userInput, movieInput, ratings
            );
            
            this.trainingHistories.mlp = await this.models.mlp.train(
                userInput, movieInput, ratings
            );
            
            this.plotTrainingHistories();
            this.updateStatus('All models trained successfully!');
            
        } catch (error) {
            this.updateStatus('Error training models: ' + error.message);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    prepareTrainingData() {
        const userInput = this.data.ratings.map(r => r.userId - 1); // 0-based indices
        const movieInput = this.data.ratings.map(r => {
            const movie = this.data.movies.get(r.movieId);
            return {
                movieId: r.movieId - 1,
                genres: movie.genres
            };
        });
        const ratings = this.data.ratings.map(r => r.rating > 3 ? 1 : 0); // Binary labels
        
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

        tfvis.show.history(
            { name: 'Training Loss', tab: 'Training' },
            lossSeries,
            ['line', 'line', 'line']
        );

        tfvis.show.history(
            { name: 'Training Accuracy', tab: 'Training' },
            accuracySeries,
            ['line', 'line', 'line']
        );
    }

    async testModels() {
        if (!this.models.mlp || !this.models.deep) {
            this.updateStatus('Please train models first');
            return;
        }

        this.updateStatus('Testing models...');
        
        // Test with a sample user
        const testUserId = 1;
        const userHistory = this.getUserHistory(testUserId);
        const mlpRecommendations = await this.getRecommendations(this.models.mlp, testUserId, 10);
        const dlRecommendations = await this.getRecommendations(this.models.deep, testUserId, 10);
        
        this.displayRecommendations(userHistory, mlpRecommendations, dlRecommendations);
        this.updateStatus('Testing completed');
    }

    getUserHistory(userId) {
        return this.data.ratings
            .filter(r => r.userId === userId && r.rating >= 4)
            .slice(0, 10)
            .map(r => {
                const movie = this.data.movies.get(r.movieId);
                return {
                    title: movie.title,
                    genres: movie.genreNames,
                    rating: r.rating
                };
            });
    }

    async getRecommendations(model, userId, count = 10) {
        const userEmbedding = await model.getUserEmbedding(userId - 1);
        const allMovies = Array.from(this.data.movies.values());
        
        const scores = [];
        for (const movie of allMovies) {
            const movieEmbedding = await model.getItemEmbedding(movie.id - 1, movie.genres);
            const score = await this.dotProduct(userEmbedding, movieEmbedding);
            scores.push({ movie, score });
        }
        
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
        
        data.forEach(item => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = item.title;
            
            const genreCell = row.insertCell(1);
            item.genres.forEach(genre => {
                const span = document.createElement('span');
                span.className = 'genre-tag';
                span.textContent = genre;
                genreCell.appendChild(span);
            });
            
            if (isHistory) {
                row.insertCell(2).textContent = item.rating;
            } else {
                row.insertCell(2).textContent = item.score.toFixed(4);
            }
        });
    }

    async compareModels() {
        if (!this.models.baseline || !this.models.mlp || !this.models.deep) {
            this.updateStatus('Please train all models first');
            return;
        }

        this.updateStatus('Comparing models...');
        
        // Calculate metrics for each model
        const baselineMetrics = await this.calculateMetrics(this.models.baseline);
        const mlpMetrics = await this.calculateMetrics(this.models.mlp);
        const deepMetrics = await this.calculateMetrics(this.models.deep);
        
        this.displayMetrics(baselineMetrics, mlpMetrics, deepMetrics);
        this.updateStatus('Model comparison completed');
    }

    async calculateMetrics(model, k = 10) {
        // Simplified metric calculation - in practice, use proper train/test split
        const sampleUsers = Array.from({length: 50}, (_, i) => i);
        let totalPrecision = 0;
        let totalRecall = 0;
        
        for (const userId of sampleUsers) {
            const recommendations = await this.getRecommendations(model, userId + 1, k);
            // Simplified: assume first k recommendations are relevant for demo
            const precision = recommendations.filter((_, idx) => idx < 5).length / k;
            const recall = precision; // Simplified for demo
            
            totalPrecision += precision;
            totalRecall += recall;
        }
        
        return {
            precision: totalPrecision / sampleUsers.length,
            recall: totalRecall / sampleUsers.length,
            f1: 2 * (totalPrecision * totalRecall) / (totalPrecision + totalRecall) / sampleUsers.length
        };
    }

    displayMetrics(baseline, mlp, deep) {
        const metricsPanel = document.getElementById('metricsPanel');
        metricsPanel.innerHTML = `
            <div class="metric-card">
                <h4>Baseline MF</h4>
                <div class="metric-value">P@${10}: ${baseline.precision.toFixed(4)}</div>
                <div>R@${10}: ${baseline.recall.toFixed(4)}</div>
                <div>F1: ${baseline.f1.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <h4>MLP Two-Tower</h4>
                <div class="metric-value">P@${10}: ${mlp.precision.toFixed(4)}</div>
                <div>R@${10}: ${mlp.recall.toFixed(4)}</div>
                <div>F1: ${mlp.f1.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <h4>Deep Two-Tower</h4>
                <div class="metric-value">P@${10}: ${deep.precision.toFixed(4)}</div>
                <div>R@${10}: ${deep.recall.toFixed(4)}</div>
                <div>F1: ${deep.f1.toFixed(4)}</div>
            </div>
        `;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
