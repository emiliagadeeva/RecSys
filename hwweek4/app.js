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
            
            const movieCount = this.data.movies ? this.data.movies.size : 0;
            const ratingCount = this.data.ratings ? this.data.ratings.length : 0;
            const userCount = this.data.users ? this.data.users.size : 0;
            
            this.updateStatus(`✅ Data loaded successfully: ${ratingCount} ratings, ${movieCount} movies, ${userCount} users`);
            
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
        console.log('Created mock movies data');
    }

    async loadMockRatings() {
        const ratings = [];
        
        // Create more realistic mock ratings
        for (let i = 0; i < 5000; i++) { // Smaller dataset for performance
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
        console.log('Created mock ratings data');
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
        console.log('Created mock users data');
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
        await new Promise(resolve => setTimeout(resolve, 50));
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
            // Initialize models with smaller embedding dimension for stability
            this.models = {
                baseline: new BaselineTwoTower(16, 943, 1682, 19),
                deep: new DeepLearningTwoTower(16, 943, 1682, 19),
                mlp: new MLPTwoTower(16, 943, 1682, 19)
            };
            
            // Prepare training data
            const { userInput, movieInput, ratings } = this.prepareTrainingData();
            
            // Train models with fewer epochs for demo
            this.updateStatus('📈 Training Baseline Matrix Factorization...');
            this.trainingHistories.baseline = await this.models.baseline.train(
                userInput, movieInput, ratings, 3, 64
            );
            
            this.updateStatus('🧠 Training Deep Learning Two-Tower...');
            this.trainingHistories.deep = await this.models.deep.train(
                userInput, movieInput, ratings, 3, 64
            );
            
            this.updateStatus('🔬 Training MLP Two-Tower...');
            this.trainingHistories.mlp = await this.models.mlp.train(
                userInput, movieInput, ratings, 3, 64
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
        // Use only a small subset for faster training in browser
        const subsetSize = Math.min(800, this.data.ratings.length);
        const subsetRatings = this.data.ratings.slice(0, subsetSize);
        
        const userInput = subsetRatings.map(r => r.userId - 1); // 0-based indices
        const movieInput = subsetRatings.map(r => {
            const movie = this.data.movies.get(r.movieId);
            return {
                movieId: r.movieId - 1,
                genres: movie.genres
            };
        });
        
        // Convert ratings to binary labels (1 for ratings >= 4, 0 otherwise)
        const ratings = subsetRatings.map(r => r.rating >= 4 ? 1 : 0);
        
        console.log(`Training data: ${userInput.length} samples, ${ratings.filter(r => r === 1).length} positive samples`);
        
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

        // Filter out models that failed to train
        const validHistories = histories.filter(model => model.history && model.history.loss.length > 0);
        
        if (validHistories.length === 0) {
            this.updateStatus('❌ No models trained successfully');
            return;
        }

        const lossSeries = validHistories.map(model => ({
            values: model.history.loss,
            name: model.name
        }));

        const accuracySeries = validHistories.map(model => ({
            values: model.history.acc,
            name: model.name
        }));

        // Clear previous charts
        const chartsContainer = document.getElementById('trainingCharts');
        chartsContainer.innerHTML = '<h3>📊 Training Progress</h3>';

        // Create loss chart
        const lossTab = { name: 'Training Loss', tab: 'Training' };
        tfvis.show.history(
            lossTab,
            lossSeries,
            ['line', 'line', 'line'].slice(0, validHistories.length),
            {
                xLabel: 'Epoch',
                yLabel: 'Loss',
                width: 450,
                height: 300
            },
            chartsContainer
        );

        // Create accuracy chart
        const accTab = { name: 'Training Accuracy', tab: 'Training' };
        tfvis.show.history(
            accTab,
            accuracySeries,
            ['line', 'line', 'line'].slice(0, validHistories.length),
            {
                xLabel: 'Epoch',
                yLabel: 'Accuracy',
                width: 450,
                height: 300
            },
            chartsContainer
        );
    }

    async testModels() {
        if (!this.models.mlp || !this.models.deep) {
            this.updateStatus('❌ Please train models first');
            return;
        }

        this.updateStatus('🧪 Testing models with sample user...');
        
        // Find a user with sufficient rating history
        const testUserId = this.findUserWithRatings();
        if (!testUserId) {
            this.updateStatus('❌ No users with sufficient rating history found');
            return;
        }
        
        try {
            const userHistory = this.getUserHistory(testUserId);
            const mlpRecommendations = await this.getRecommendations(this.models.mlp, testUserId, 10);
            const dlRecommendations = await this.getRecommendations(this.models.deep, testUserId, 10);
            
            this.displayRecommendations(userHistory, mlpRecommendations, dlRecommendations);
            this.updateStatus('✅ Testing completed. Check comparison section below.');
        } catch (error) {
            this.updateStatus('❌ Error during testing: ' + error.message);
            console.error('Testing error:', error);
        }
    }

    getUserHistory(userId) {
        const userRatings = this.data.ratings
            .filter(r => r.userId === userId)
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 10);
        
        return userRatings.map(r => {
            const movie = this.data.movies.get(r.movieId);
            return {
                title: movie ? movie.title : `Movie ${r.movieId}`,
                genres: movie ? movie.genreNames : ['Unknown'],
                rating: r.rating
            };
        });
    }

    findUserWithRatings() {
        // Find a user with at least 3 ratings of 4+ stars
        const userRatingCounts = new Map();
        
        this.data.ratings.forEach(rating => {
            if (rating.rating >= 4) {
                userRatingCounts.set(rating.userId, (userRatingCounts.get(rating.userId) || 0) + 1);
            }
        });
        
        for (const [userId, count] of userRatingCounts) {
            if (count >= 3) {
                return userId;
            }
        }
        
        // Fallback to any user with ratings
        const usersWithRatings = new Set(this.data.ratings.map(r => r.userId));
        return usersWithRatings.size > 0 ? Array.from(usersWithRatings)[0] : null;
    }

    async getRecommendations(model, userId, count = 10) {
        this.updateStatus(`🔍 Generating recommendations for user ${userId}...`);
        
        try {
            const userEmbedding = await model.getUserEmbedding(userId - 1);
            const allMovies = Array.from(this.data.movies.values());
            
            const scores = [];
            const batchSize = 50; // Smaller batch size for stability
            
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
                
        } catch (error) {
            console.error('Error generating recommendations:', error);
            return [];
        }
    }

    async dotProduct(vec1, vec2) {
        try {
            const product = vec1.mul(vec2);
            const sum = product.sum();
            const result = await sum.data();
            return result[0];
        } finally {
            // Cleanup is handled by the caller
        }
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
            
            // Title cell
            const titleCell = row.insertCell(0);
            titleCell.textContent = item.title;
            titleCell.style.maxWidth = '200px';
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
            
            // Rating/Score cell
            const valueCell = row.insertCell(2);
            if (isHistory) {
                valueCell.textContent = '⭐'.repeat(item.rating);
                valueCell.title = `Rating: ${item.rating}/5`;
            } else {
                valueCell.textContent = item.score.toFixed(4);
                valueCell.style.fontFamily = 'monospace';
            }
        });
    }

    async compareModels() {
        if (!this.models.baseline || !this.models.mlp || !this.models.deep) {
            this.updateStatus('❌ Please train all models first');
            return;
        }

        this.updateStatus('📊 Comparing model performance...');
        
        try {
            // Calculate metrics for each model
            const baselineMetrics = await this.calculateMetrics(this.models.baseline);
            const mlpMetrics = await this.calculateMetrics(this.models.mlp);
            const deepMetrics = await this.calculateMetrics(this.models.deep);
            
            this.displayMetrics(baselineMetrics, mlpMetrics, deepMetrics);
            this.updateStatus('✅ Model comparison completed!');
        } catch (error) {
            this.updateStatus('❌ Error comparing models: ' + error.message);
            console.error('Comparison error:', error);
        }
    }

    async calculateMetrics(model, k = 5) {
        // Use a small sample of users for performance
        const sampleUsers = Array.from({length: 10}, (_, i) => i + 1);
        let totalPrecision = 0;
        let totalRecall = 0;
        let userCount = 0;
        
        for (const userId of sampleUsers) {
            try {
                const userRatings = this.data.ratings.filter(r => r.userId === userId && r.rating >= 4);
                if (userRatings.length < 2) continue; // Skip users with very few ratings
                
                const recommendations = await this.getRecommendations(model, userId, k);
                if (recommendations.length === 0) continue;
                
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
        
        const createMetricCard = (modelName, metrics, color = '#007cba') => `
            <div class="metric-card">
                <h4>${modelName}</h4>
                <div class="metric-value" style="color: ${color}">${metrics.precision.toFixed(3)}</div>
                <div><strong>Precision@5</strong></div>
                <div>Recall@5: ${metrics.recall.toFixed(3)}</div>
                <div>F1 Score: ${metrics.f1.toFixed(3)}</div>
            </div>
        `;
        
        metricsPanel.innerHTML = `
            ${createMetricCard('📈 Baseline MF', baseline, '#007cba')}
            ${createMetricCard('🔬 MLP Two-Tower', mlp, '#28a745')}
            ${createMetricCard('🧠 Deep Two-Tower', deep, '#dc3545')}
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
    console.log('MovieLens Recommendation Demo initialized');
});
