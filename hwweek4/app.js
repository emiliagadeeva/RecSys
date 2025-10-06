class MovieLensApp {
    constructor() {
        this.data = null;
        this.movies = null;
        this.users = null;
        this.models = {};
        this.trainingHistory = {};
        this.testUserId = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModels').addEventListener('click', () => this.trainAllModels());
        document.getElementById('testModels').addEventListener('click', () => this.testModels());
        document.getElementById('compareModels').addEventListener('click', () => this.compareModels());
    }

    async loadData() {
        this.updateStatus('Loading MovieLens 100K data...', 'loading');
        
        try {
            // Load ratings data from local data folder
            const ratingsResponse = await fetch('./data/u.data');
            if (!ratingsResponse.ok) throw new Error('Failed to load u.data');
            const ratingsText = await ratingsResponse.text();
            
            // Load movie data from local data folder
            const moviesResponse = await fetch('./data/u.item');
            if (!moviesResponse.ok) throw new Error('Failed to load u.item');
            const moviesText = await moviesResponse.text();
            
            // Load user data if available from local data folder
            let usersText = null;
            try {
                const usersResponse = await fetch('./data/u.user');
                if (usersResponse.ok) {
                    usersText = await usersResponse.text();
                }
            } catch (e) {
                console.log('User data not available, proceeding without it');
            }
            
            this.parseData(ratingsText, moviesText, usersText);
            this.updateStatus('Data loaded successfully!', 'success');
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`, 'error');
            console.error('Data loading error:', error);
        }
    }

    parseData(ratingsText, moviesText, usersText) {
        // Parse ratings
        const ratings = [];
        const lines = ratingsText.trim().split('\n');
        for (const line of lines) {
            const [user_id, item_id, rating, timestamp] = line.split('\t');
            ratings.push({
                user_id: parseInt(user_id),
                item_id: parseInt(item_id),
                rating: parseFloat(rating),
                timestamp: parseInt(timestamp)
            });
        }

        // Parse movies and genres
        const movies = {};
        const genreList = [
            'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ];
        
        const movieLines = moviesText.trim().split('\n');
        for (const line of movieLines) {
            const parts = line.split('|');
            if (parts.length < 5) continue;
            
            const item_id = parseInt(parts[0]);
            const title = parts[1];
            // Parse genre flags (positions 5-23)
            const genreFlags = [];
            for (let i = 5; i < Math.min(parts.length, 24); i++) {
                genreFlags.push(parseInt(parts[i]) || 0);
            }
            // Pad if necessary
            while (genreFlags.length < 19) {
                genreFlags.push(0);
            }
            
            const genres = genreFlags.map((flag, index) => flag === 1 ? genreList[index] : null)
                                   .filter(genre => genre !== null);
            
            movies[item_id] = {
                id: item_id,
                title: title,
                genres: genres,
                genreFeatures: genreFlags
            };
        }

        // Parse users if available
        const users = {};
        if (usersText) {
            const userLines = usersText.trim().split('\n');
            for (const line of userLines) {
                const parts = line.split('|');
                if (parts.length >= 4) {
                    const [user_id, age, gender, occupation, zip_code] = parts;
                    users[parseInt(user_id)] = {
                        age: parseInt(age),
                        gender: gender,
                        occupation: occupation,
                        zip_code: zip_code
                    };
                }
            }
        }

        this.data = {
            ratings: ratings,
            movies: movies,
            users: users,
            numUsers: Math.max(...ratings.map(r => r.user_id)),
            numItems: Math.max(...ratings.map(r => r.item_id))
        };

        console.log(`Loaded ${ratings.length} ratings, ${Object.keys(movies).length} movies, ${Object.keys(users).length} users`);
    }

    async trainAllModels() {
        if (!this.data) {
            this.updateStatus('Please load data first!', 'error');
            return;
        }

        this.updateStatus('Training all models...', 'loading');

        // Initialize models
        this.models = {
            noDL: new WithoutDLTwoTower(this.data.numUsers, this.data.numItems, 32, 19),
            mlp: new MLPTwoTower(this.data.numUsers, this.data.numItems, 32, 19, this.data.users),
            deep: new DeepLearningTwoTower(this.data.numUsers, this.data.numItems, 32, 19, this.data.users)
        };

        this.trainingHistory = {};

        // Train each model
        for (const [modelName, model] of Object.entries(this.models)) {
            this.updateStatus(`Training ${modelName}...`, 'loading');
            
            try {
                const history = await model.train(this.data.ratings, this.data.movies, 5, 64);
                this.trainingHistory[modelName] = history;
                this.updateChart(modelName, history);
            } catch (error) {
                console.error(`Error training ${modelName}:`, error);
                this.updateStatus(`Error training ${modelName}: ${error.message}`, 'error');
            }
        }

        this.updateStatus('All models trained successfully!', 'success');
    }

    updateChart(modelName, history) {
        const chartData = {
            values: history.loss.map((loss, epoch) => ({ epoch: epoch + 1, loss: loss }))
        };

        const spec = {
            $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
            data: chartData,
            width: 300,
            height: 200,
            mark: 'line',
            encoding: {
                x: { field: 'epoch', type: 'quantitative', title: 'Epoch' },
                y: { field: 'loss', type: 'quantitative', title: 'Loss' }
            },
            title: `Training Loss - ${modelName}`
        };

        const chartId = `chart${modelName.charAt(0).toUpperCase() + modelName.slice(1)}`;
        vegaEmbed(`#${chartId}`, spec);
    }

    async testModels() {
        if (!this.models.noDL || !this.models.noDL.model) {
            this.updateStatus('Please train models first!', 'error');
            return;
        }

        this.updateStatus('Testing models...', 'loading');

        // Select a random user for testing (ensure user exists in data)
        const userRatingsMap = {};
        this.data.ratings.forEach(r => {
            if (!userRatingsMap[r.user_id]) userRatingsMap[r.user_id] = [];
            userRatingsMap[r.user_id].push(r);
        });
        
        const availableUsers = Object.keys(userRatingsMap).map(Number);
        const userId = availableUsers[Math.floor(Math.random() * availableUsers.length)];
        this.testUserId = userId;

        // Get user's historical ratings
        const userRatings = userRatingsMap[userId]
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 10)
            .map(r => ({
                title: this.data.movies[r.item_id]?.title || `Movie ${r.item_id}`,
                rating: r.rating,
                genres: this.data.movies[r.item_id]?.genres?.join(', ') || 'Unknown'
            }));

        // Display user history
        this.displayUserHistory(userRatings);

        // Get recommendations from each model
        const recommendations = {};
        for (const [modelName, model] of Object.entries(this.models)) {
            try {
                const userRecs = await model.recommend(userId, this.data.movies, 10);
                recommendations[modelName] = userRecs;
                this.displayRecommendations(modelName, userRecs);
            } catch (error) {
                console.error(`Error getting recommendations from ${modelName}:`, error);
                recommendations[modelName] = [];
            }
        }

        // Show comparison section
        document.getElementById('modelsComparison').style.display = 'flex';
        this.updateStatus(`Testing completed for user ${userId}`, 'success');
    }

    displayUserHistory(ratings) {
        const tbody = document.querySelector('#userHistoryTable tbody');
        tbody.innerHTML = '';
        
        ratings.forEach(movie => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = movie.title;
            row.insertCell(1).textContent = movie.rating;
            row.insertCell(2).textContent = movie.genres;
        });
        
        document.getElementById('userHistory').style.display = 'block';
    }

    displayRecommendations(modelName, recommendations) {
        const tableId = `table${modelName.charAt(0).toUpperCase() + modelName.slice(1)}`;
        const tbody = document.querySelector(`#${tableId} tbody`);
        tbody.innerHTML = '';
        
        recommendations.forEach(movie => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = movie.title;
            row.insertCell(1).textContent = movie.score.toFixed(4);
            row.insertCell(2).textContent = movie.genres.join(', ');
        });
    }

    async compareModels() {
        if (!this.models.noDL || !this.testUserId) {
            this.updateStatus('Please train and test models first!', 'error');
            return;
        }

        this.updateStatus('Comparing models...', 'loading');

        const metrics = {};
        const k = 10;

        for (const [modelName, model] of Object.entries(this.models)) {
            try {
                metrics[modelName] = await this.calculateMetrics(model, this.testUserId, k);
            } catch (error) {
                console.error(`Error calculating metrics for ${modelName}:`, error);
                metrics[modelName] = { precision: 0, recall: 0, hits: 0, totalPositives: 0 };
            }
        }

        this.displayMetrics(metrics);
        document.getElementById('metricsComparison').style.display = 'flex';
        this.updateStatus('Model comparison completed!', 'success');
    }

    async calculateMetrics(model, userId, k) {
        // Get user's actual positive interactions (ratings >= 4)
        const userPositives = this.data.ratings
            .filter(r => r.user_id === userId && r.rating >= 4)
            .map(r => r.item_id);

        // Get recommendations
        const recommendations = await model.recommend(userId, this.data.movies, 100);
        const recommendedIds = recommendations.map(r => r.id).slice(0, k);

        // Calculate metrics
        const hits = recommendedIds.filter(id => userPositives.includes(id)).length;
        const precision = k > 0 ? hits / k : 0;
        const recall = userPositives.length > 0 ? hits / userPositives.length : 0;

        return {
            precision: precision,
            recall: recall,
            hits: hits,
            totalPositives: userPositives.length
        };
    }

    displayMetrics(metrics) {
        const container = document.getElementById('metricsComparison');
        container.innerHTML = '<h2>Model Performance Metrics (Precision@10, Recall@10)</h2>';
        
        const metricsDiv = document.createElement('div');
        metricsDiv.className = 'metrics';
        
        for (const [modelName, metric] of Object.entries(metrics)) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <h3>${modelName}</h3>
                <p><strong>Precision@10:</strong> ${metric.precision.toFixed(4)}</p>
                <p><strong>Recall@10:</strong> ${metric.recall.toFixed(4)}</p>
                <p><strong>Hits:</strong> ${metric.hits}/${metric.totalPositives}</p>
            `;
            metricsDiv.appendChild(card);
        }
        
        container.appendChild(metricsDiv);
    }

    updateStatus(message, type = 'info') {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = type;
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
